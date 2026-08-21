import numpy as np

def skew(v):
    x,y,z=np.asarray(v,float); return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
def so3_exp(v):
    v=np.asarray(v,float); a=np.linalg.norm(v); K=skew(v)
    if a<1e-8: return np.eye(3)+K+0.5*K@K
    return np.eye(3)+np.sin(a)/a*K+(1-np.cos(a))/a**2*K@K
def so3_log(R):
    R=np.asarray(R,float); a=np.arccos(np.clip((np.trace(R)-1)/2,-1,1))
    q=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
    return 0.5*q if a<1e-8 else a*q/(2*np.sin(a))
def quat_to_rot(q):
    x,y,z,w=np.asarray(q,float); n=np.linalg.norm([x,y,z,w]); x,y,z,w=np.array([x,y,z,w])/n
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def rot_to_euler(R):
    return np.array([np.arctan2(R[2,1],R[2,2]),np.arcsin(np.clip(-R[2,0],-1,1)),np.arctan2(R[1,0],R[0,0])])
def enu_to_ned(v):
    v=np.asarray(v,float); return np.array([v[1],v[0],-v[2]])
def yaw_enu_to_ned(yaw): return np.pi/2-yaw
