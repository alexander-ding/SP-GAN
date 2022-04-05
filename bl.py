import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import smplx
import scipy

import os
import json

from hier_execute import *
import tqdm
from mesh_intersection.bvh_search_tree import BVH
import mesh_intersection.loss as collisions_loss
import pickle
from prox import isvalid
import stability

from sdf_from_pcd import get_padded_bounds

device = torch.device("cuda")

body_segments_dir = f'prox/data/body_segments'
body_parts = ['back','gluteus','thighs']
contact_faces = []
for part in body_parts:
    with open(os.path.join(body_segments_dir, part + '.json'), 'r') as f:
        data = json.load(f)
        arr = list(data["faces_ind"])
        arr = torch.tensor(np.unique(arr)).to(device)
        contact_faces.append(arr)

with open("vol_dict.npz","rb") as f:
    vols = torch.Tensor(pickle.load(f)).to(device)

pose_pair,pose_solo = np.array([
    [0,1],
    [3,4],
    [6,7],
    [9,10],
    [12,13],
    [15,16],
    [17,18],
    [19,20]
]).T,\
np.array([
    2,5,8,11,14
])

dim = 12000

search_tree = BVH(max_collisions=8)
pen_distance = \
    collisions_loss.DistanceFieldPenetrationLoss(
        sigma=0.5, point2plane=False,
        vectorized=True, penalize_outside=True)

init_pose = torch.load('init_pose.pt').to(device)
faces = torch.load('faces.pt').to(device)
back_faces = faces[contact_faces[0]]
glutes_faces = faces[torch.cat((contact_faces[1],contact_faces[2]))]

"""
from modelDecoder import Decoder
im_decoder = Decoder(256,3,128).to(device)
im_decoder.load_state_dict(torch.load('modelD_dict.pt'))
for param in im_decoder.parameters():
    param.requires_grad = False
im_decoder.eval()
"""
def random_pose():

    return mean + torch.randn(std.shape).to(device) * std

def make_model(transl=None,orient=None,pose=None,betas=None,bsz=1,full_pose=False):

    npz_bdata_path = f'prox/data/Subject_3_F_11_poses.npz' # the path to body data

    bdata = np.load(npz_bdata_path)
    fId = 700

    model_dir = f"prox/data"

    if full_pose == True:
        orient = pose[:,:3]
        transl = pose[:,3:6]
        pose = pose[:,6:]
    
    if orient is None:
        orient = torch.tensor([-.75,0,0]).view(1,3).float()
    if transl is None:
        transl = torch.tensor([0,1.,.4]).view(1,3).float()
    if pose is None:
        pose = torch.Tensor(bdata['poses'][fId:fId+1, 3:66]) # controls the body
        pose[:,0] -= 0.05
        pose[:,[9,12]] = 0.
    if betas is None:
        betas = torch.tensor(bdata['betas']).view(1,-1).float() # controls the body shape
    betas = betas.view(-1,16)[:,:10]
    
    model_params = dict(model_path=model_dir,
                        create_betas=True,
                        betas=betas,
                        create_global_orient=True,
                        global_orient=orient,
                        create_body_pose=True,
                        body_pose=pose,
                        create_transl=True,
                        transl=transl,
                        batch_size=bsz,
                        dtype=torch.float32)

    model = smplx.create(gender='male',model_type="smplx", **model_params).to(device)
    model.requires_grad = True

    return model

def setup_sdf(inputs,ctype):

    if ctype == 'cuboid':
        all_eqs = []
        
        for chair in inputs:
            vs = []
            centers = []

            for cube in chair:
                v,f = cube.getTris()
                
                vs.append(v[f])
                centers.append(torch.mean(v,dim=0))

            vs = torch.stack(vs).to(device)
            centers = torch.stack(centers).to(device)
            
            a,b,c = vs.permute(2,0,1,3)
            ns = torch.cross(c-b,b-a,dim=-1)
            size = torch.norm(ns.clone(),dim=-1,keepdim=True)
            ns /= size
            
            d = -((a + b + c)/3. * ns.clone()).sum(-1)
        
            dots = (centers.unsqueeze(1) * ns).sum(-1)
            idx = (dots + d > 0.).nonzero()
            
            ns[idx[:,0],idx[:,1]] *= -1
            d[idx[:,0],idx[:,1]] *= -1

            eqs = torch.cat((ns,d.unsqueeze(-1)),dim=-1).permute(2,0,1)
            all_eqs.append(eqs)
            
        return lambda pts: compute_sdf_cubes(all_eqs,pts)

    elif ctype == 'mesh':
        
        from sdf_from_pcd import compute_sdf_pcd
        sdfs, grid_mins, grid_maxs = inputs
        grid_mins, grid_maxs = get_padded_bounds((grid_mins, grid_maxs))
        return lambda pts: compute_sdf_pcd(sdfs, grid_mins, grid_maxs, pts)

    grids,box_min,box_max = [],[],[]
    for item in inputs:
        grids.append(item['sdf'])
        box_min.append(item['grid_min'])
        box_max.append(item['grid_max'])
    grids = torch.stack(grids).to(device)
    box_min = torch.stack(box_min).to(device).unsqueeze(1)
    box_max = torch.stack(box_max).to(device).unsqueeze(1)
    
    return lambda pts: compute_sdf_sdf(grids,box_min,box_max,pts)

def compute_sdf_cubes(all_eqs,pts):

    sdfs = []
    for i in range(len(pts)):

        p = pts[i]
        eqs = all_eqs[i % len(all_eqs)]

        ones = torch.ones(p.shape[:-1]).unsqueeze(-1).to(device)
        p = torch.cat((p,ones),dim=-1).to(device)

        sdf_faces = torch.tensordot(p,eqs,[[-1],[0]])
        sdf_cubes = torch.max(sdf_faces,dim=-1)[0]

        sdf = torch.min(sdf_cubes,dim=-1)[0]
        
        sdfs.append(sdf)
        
    return torch.stack(sdfs)

def compute_sdf_sdf(grids,box_min,box_max,pts):
    rep = len(pts) // len(grids)
    grids = grids.repeat(rep,1,1,1,1)

    scaled_pts = (pts - box_min) / (box_max - box_min) * 2 - 1
    scaled_pts = scaled_pts.unsqueeze(2).unsqueeze(3)
    
    sdf = F.grid_sample(grids,scaled_pts).view(len(pts),-1)
    return sdf

def sample_mesh_surface(v,f,n_s=10000,seed=None):

    squeeze = False
    if len(v.shape) == 2:
        squeeze = True
        v.unsqueeze(0)
    bsz = len(v)
        
    a,b,c = v[:,f].permute(2,0,1,3)
    areas = torch.cross(a - b, b - c).norm(dim=-1) * .5
    weights = (areas / areas.sum(dim=-1,keepdim=True)).detach().cpu().numpy()

    choices = []
    np.random.seed(seed)
    for i in range(len(weights)):
        w = weights[i]
        ch = np.random.choice(a=len(w),size=n_s,p=w)
        idx = np.ones_like(ch) * i
        choices.append(np.stack((idx,ch)))
    choices = np.concatenate(choices,axis=-1)
    np.random.seed(None)
    
    u,v = torch.rand(size=(2,bsz,n_s)).to(device)
    
    pts = (1 - u**.5).view(bsz,-1,1) * a[choices[0],choices[1]].view(bsz,-1,3)
    pts += (u**.5 * (1 - v)).view(bsz,-1,1) * b[choices[0],choices[1]].view(bsz,-1,3)
    pts += (v * u ** .5).view(bsz,-1,1) * c[choices[0],choices[1]].view(bsz,-1,3)
    
    return pts

def get_gravity_loss(floor,bones,vols):

    g_loss = 9.8 * ((bones[:,:,1] - floor) * vols).abs().sum()
    return 2. * g_loss

def get_penetration_loss(vertices,get_sdf):

    limit = 0.
    sdf = get_sdf(vertices)
    
    p_loss = (sdf[sdf < limit] - limit).abs().sum()
    return w_p * p_loss

def get_self_penetration_loss(vertices, faces, search_tree, pen_distance):
    sp_loss = 0.
    for j in range(len(vertices)):
        triangles = vertices[j,faces].view(1,-1,3,3)

        collision_idxs = search_tree(triangles)

        if collision_idxs.ge(0).sum().item() > 0:
            sp_loss += torch.sum(pen_distance(triangles, collision_idxs))
            
    return 10. * sp_loss
        
def get_valid_loss(bones):

    valid_loss = 0.
    S = bones.permute(0,2,1)
    valid_loss = isvalid.invalid_pose_loss(S).sum()
    return 0.1 * valid_loss

def get_symmetry_loss(pose,orient,transl,pose_pair,pose_solo,x_avg):

    sym_loss = 0.
    # Pair symmetry
    side_a,side_b = pose[:,pose_pair].permute(1,0,2,3)
    side_b *= torch.Tensor([1.,-1.,-1.]).to(device)
    sym_loss += (side_a-side_b).norm(dim=-1).sum()
    # Solo symmetry
    sym_loss += pose[:,pose_solo,[[1],[2]]].abs().sum()
    sym_loss += orient[:,[1,2]].abs().sum()
    sym_loss += (transl[:,0] - x_avg).abs().sum() * 10.
    return 2.5 * sym_loss

def get_spine_loss(pose,init_pose):

    # Neck                                                                                          
    neck_loss = pose[:,[11,14],0].abs().sum()

    # Lumbar
    lumbar_loss = (pose[:,[2,5],0] - init_pose[[2,5],0]).abs().sum()

    # Lumbar2
    lumbar2_loss = (pose[:,8,0] - init_pose[8,0]).abs().sum()
        
    return 5. * neck_loss + 5. * lumbar_loss + lumbar2_loss

def get_contact_loss(vertices,dim,back_faces,glutes_faces,get_sdf):

    limit = 0.
    num = int(dim*0.05)

    # Back contact
    points = sample_mesh_surface(vertices,back_faces,n_s=dim)
    sdf = get_sdf(points).abs()
    c_loss_back = (sdf[sdf > limit] - limit).sum() / dim
    # Glutes contact
    points = sample_mesh_surface(vertices,glutes_faces,n_s=dim)
    sdf = get_sdf(points).abs()
    c_loss_glutes = (sdf[sdf > limit] - limit).sum() / dim
    # Total contact                                                                             
    c_loss = c_loss_back + c_loss_glutes
    return w_c * c_loss

def get_loss(model,batch_size,faces,box_min,get_sdf,x_avg,init_pose):
    output = model(return_vertices=True)
    
    pose = output.body_pose.view(batch_size,-1,3)
    vertices = output.vertices
    pts = sample_mesh_surface(vertices,faces)
    joints = output.joints
    bones = joints[:,joints_idx()]
    
    # Gravity
    gravity_loss = get_gravity_loss(box_min[:,1].unsqueeze(1),bones,vols)
    g_loss = 9.8 * ((bones[:,:,2] - box_min[:,2].unsqueeze(1)) * vols).abs().sum()
    gravity_loss += g_loss
    
    # Self Penetration
    self_penetration_loss = get_self_penetration_loss(vertices,faces,search_tree,pen_distance)
    
    # Penetration
    penetration_loss = get_penetration_loss(pts,get_sdf)
    
    # Invalid Pose
    valid_loss = get_valid_loss(bones)
    
    # Symmetry
    symmetry_loss = get_symmetry_loss(pose,model.global_orient,model.transl,pose_pair,pose_solo,x_avg)
    
    # Spine Loss
    spine_loss = get_spine_loss(pose,init_pose)
    
    # Contact Loss
    contact_loss = get_contact_loss(vertices,dim,back_faces,glutes_faces,get_sdf)
    
    loss_items = [
        gravity_loss,
        self_penetration_loss,
        penetration_loss,
        valid_loss,
        symmetry_loss,
        spine_loss,
        contact_loss
    ]
    
    loss = 0.
    for item in loss_items:
        if item > 0.:
            loss += item

    return loss

def setup_simulation(inputs,betas,poses,ctype):

    global w_p,w_c
    num_betas = len(betas) if betas is not None else 1
    
    if ctype == 'cuboid':
        # Weights
        w_p = 1.3
        w_c = 17.5
        # Batching
        if isinstance(inputs[0],list):
            num_chairs = len(inputs)
        else:
            num_chairs = 1
            inputs = [inputs]
        # BBox
        box_min = torch.stack([
            stability.get_corners(cbs).min(dim=0)[0] for cbs in inputs
        ])
        box_max = torch.stack([
            stability.get_corners(cbs).max(dim=0)[0] for cbs in inputs
        ])
    elif ctype == 'mesh':
        #Weights
        w_p = 1.3
        w_c = 30.
        sdfs, box_min, box_max = inputs
        num_chairs = len(sdfs)
    else:
        # Weights
        w_p = 1.3e-2
        w_c = 1e-2
        num_chairs = len(inputs)
        # BBox
        box_min,box_max = [],[]
        for item in inputs:
            box_min.append(item['grid_min'])
            box_max.append(item['grid_max'])
        box_min,box_max = torch.stack(box_min).to(device),torch.stack(box_max).to(device)

    get_sdf = setup_sdf(inputs,ctype)

    batch_size = max(num_chairs,num_betas)

    # Initialize Model
    if poses is None:
        start_y = box_max[:,1] + 0.15
        start_z = (box_max[:,2] + box_min[:,2]) / 2.

        transl = torch.stack([torch.zeros(start_y.shape),start_y.cpu(),start_z.cpu()]).T.float()
        pose = None
        orient = None
    else:
        orient = poses[:,:3]
        transl = poses[:,3:6]
        pose = poses[:,6:]
    x_avg = (box_max[:,0] + box_min[:,0]) / 2.

        
    model = make_model(transl=transl,orient=orient,pose=pose,betas=betas,bsz=batch_size)
    
    return model,get_sdf,box_min,box_max,batch_size,x_avg
    
def simulation(inputs,betas=None,ctype='cuboid',progbar=False):

    poses = None
    model,get_sdf,box_min,box_max,batch_size,x_avg = setup_simulation(inputs,betas,poses,ctype)

    range_a = 300
    # Run Simulation Optimization
    if progbar:
        pbar = tqdm.tqdm(total=300)

    params = [model.transl,model.global_orient,model.body_pose]
    opt = torch.optim.Adam(params, lr = 3e-3, eps = 1e-4)
    
    torch.autograd.set_detect_anomaly(True)

    u = []
    for i in range(range_a):
        loss =  get_loss(model,batch_size,faces,box_min,get_sdf,x_avg,init_pose)
        
        if loss > 0.:
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

        if i % 50 == 0:
            vertices = model(return_vertices=True).vertices
            u.append((get_sdf(vertices),vertices))
        del loss
        
        if progbar:
            pbar.update(1)
    torch.save(u,'temp.pt')
    
    return torch.cat((model.global_orient,model.transl,model.body_pose),dim=-1).cpu().detach().numpy()

def more_pose(inputs,poses,betas,ctype='cuboid'):

    model,get_sdf,box_min,box_max,batch_size,x_avg = setup_simulation(inputs,betas,poses,ctype)

    params = [model.transl,model.global_orient,model.body_pose]
    opt = torch.optim.Adam(params, lr = 2e-3)

    for i in range(10):
        loss =  get_loss(model,batch_size,faces,box_min,get_sdf,x_avg,init_pose)
        
        if loss > 0.:
            opt.zero_grad()
            loss.backward()
            opt.step()

        del loss

        return torch.cat((model.global_orient,model.transl,model.body_pose),dim=-1).cpu().detach().numpy()
    
def pose_to_loss(pose, inputs, betas=None, ctype='cuboid'):

    if ctype == 'cuboid':
        if isinstance(inputs[0],list):
            num_chairs = len(inputs)
        else:
            num_chairs = 1
            inputs = [inputs]
    elif ctype == 'pcd':
        num_chairs = len(inputs)
    else:
        num_chairs = len(inputs)
    # SDF fn
    get_sdf = setup_sdf(inputs,ctype)
            
    if betas is not None:
        num_betas = len(betas)
    else:
        num_betas = 1
    batch_size = max(num_chairs,num_betas)

    if not torch.is_tensor(pose):
        pose = torch.Tensor(pose)

    model = make_model(pose=pose,betas=betas,full_pose=True)
    faces = torch.tensor(model.faces.astype(np.long)).to(device)
    part_faces = [faces[part] for part in contact_faces]

    # Contact
    output = model()
    vertices = output.vertices.to(device)
    dim = 12000
    
    losses = np.zeros(shape=batch_size)
    for j in range(3):
        
        f = part_faces[j]
        
        points = sample_mesh_surface(vertices,f,n_s=dim)  
        
        sdf = get_sdf(points).abs()
        for i in range(len(sdf)):
             losses[i] += float(sdf[sdf > 1e-3].sum().detach().cpu().numpy())

    return losses
def pts_joints_voronoi(pose,betas,batch_size):

    # Assign points in t-pose
    model_zero = make_model(pose=torch.zeros_like(pose),betas=betas,bsz=batch_size,full_pose=True)

    output = model_zero(return_vertices=True)
    vertices = output.vertices.detach()
    joints = output.joints[:,joints_idx()].detach().cpu().numpy()
    faces = torch.tensor(model_zero.faces.astype(np.long)).to(device)

    seed = np.random.randint(1)
    pts_zero = sample_mesh_surface(vertices,faces,seed=seed).cpu().numpy()

    idxs = []
    for i in range(len(pts_zero)):
        idx = scipy.spatial.distance.cdist(pts_zero[i],joints[i]).argmin(axis=-1)
        idxs.append(idx)
    idxs = np.array(idxs)
        
    # Determine points in real pose
    model = make_model(pose=pose,betas=betas,full_pose=True)
    
    output = model_zero(return_vertices=True)
    vertices = output.vertices.detach()

    pts = sample_mesh_surface(vertices,faces,seed=seed)
    return pts, idxs
    
def pose_to_loss_v2(pose, inputs, betas=None, ctype='cuboid'):

    if ctype == 'cuboid':
        if isinstance(inputs[0], list):
            num_chairs = len(inputs)
        else:
            num_chairs = 1
            inputs = [inputs]
    elif ctype == 'mesh':
        sdfs, grid_mins, grid_maxs = inputs
        num_chairs = len(sdfs)
    else:
        num_chairs = len(inputs)
    # SDF fn
    get_sdf = setup_sdf(inputs, ctype)
            
    if betas is not None:
        num_betas = len(betas)
    else:
        num_betas = 1
    batch_size = max(num_chairs, num_betas)

    if not torch.is_tensor(pose):
        pose = torch.Tensor(pose)

    # Contact
    with torch.no_grad():
        pts,pts_idx = pts_joints_voronoi(pose,betas,batch_size)
        
        sdf = get_sdf(pts).view(batch_size,-1).detach().cpu().numpy()

    losses = []
    for i in range(batch_size):
        contact_idx = pts_idx[i,sdf[i] < 0.1]

        vals,counts = np.unique(contact_idx,return_counts=True)
        
        area = np.zeros(joints_idx().shape[0])
        area[vals] = counts / len(pts_idx[i])
        pressure = (vols[area != 0.].cpu().numpy() / area[area != 0.])

        l = pressure.sum() / vols[area != 0.].sum().cpu().numpy()
        losses.append(l)

    return losses


def get_vf_model(full_pose):
    full_pose = torch.Tensor(full_pose)

    orient = full_pose[:,:3]
    transl = full_pose[:,3:6]
    pose = full_pose[:,6:]

    model = make_model(orient=orient,transl=transl,pose=pose)

    b_f = torch.tensor(model.faces.astype(np.long))
    b_v = model(return_vertices=True).vertices[0].detach().cpu()

    return b_v,b_f

def export_pose_chair(full_pose,v,f,filename,betas=None):
    import trimesh
    full_pose = torch.Tensor(full_pose).view(1,-1).float()

    orient = full_pose[:,:3]
    transl = full_pose[:,3:6]
    pose = full_pose[:,6:]

    model = make_model(orient=orient,transl=transl,pose=pose,betas=betas)

    b_f = torch.tensor(model.faces.astype(np.long))
    b_v = model(return_vertices=True).vertices[0].detach().cpu()
    
    new_f = torch.cat((f,b_f+len(v)),dim=0).detach()
    new_v = torch.cat((v,b_v),dim=0).detach()
    trimesh.Trimesh(new_v,new_f).export(filename)

def export_model_chair(model,cubes,filename):
    import trimesh
    v,f = get_vf(cubes)
    
    b_f = torch.tensor(model.faces.astype(np.long))
    b_v = model(return_vertices=True).vertices[0].detach().cpu()

    new_f = torch.cat((f.cpu(),b_f+len(v)),dim=0).detach()
    new_v = torch.cat((v.cpu(),b_v),dim=0).detach()
    trimesh.Trimesh(new_v,new_f).export(filename)

def joints_idx():

    return np.array([
        0, # pelvis
        12, # neck
        16, # right shoulder
        18, # right elbow
        20, # right wrist
        17, # left shoulder
        19, # left elbow
        21, # left wrist
        15, # head
        1, # right hip
        4, # right knee
        7, # right ankle
        10, # right foot
        2, # left hip
        5, # left knee
        8, # left ankle
        11, # left foot
    ])

def pose_joints_idx():

    return np.array([
        0, # pelvis
        12, # neck
        16, # right shoulder
        18, # right elbow
        17, # left shoulder
        19, # left elbow
        15, # head
        1, # right hip
        4, # right knee
        2, # left hip
        5, # left knee
    ])

    
def get_vf(cubes):
    verts = torch.tensor([],dtype=torch.float)
    faces = torch.tensor([],dtype=torch.long)
    for cube in cubes:
        v, f = cube.getTris()
        if v is not None and f is not None:
            faces =  torch.cat((faces, (f + verts.shape[0])))
            verts = torch.cat((verts, v))
    return verts.to(device),faces.to(device)

if __name__ == "__main__":

    model = make_model()
    faces = torch.tensor(model.faces.astype(np.long))
    vertices = model(return_vertices=True).vertices[0]

    utils.writeObj(vertices,faces,"mod.obj")
