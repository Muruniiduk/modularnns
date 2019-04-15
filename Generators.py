
# coding: utf-8

# # Data with generators

# In[1]:


import pandas as pd
import numpy as np


# ## Forward movement

# In[2]:


def fmove_generator(batch_size = 64, frames = 5):
    """
    Forward movement generator. Yields x, x_vel, y, y_vel with given number of frames.
    This generator creates data with standard normal distribution.
    """
    a = np.zeros((batch_size, frames, 4))
    a[:,0,0] = np.random.randn(batch_size) #x
    a[:,:,1] = np.random.randn(batch_size).reshape(-1,1)*0.1 #x_vel
    a[:,0,2] = np.random.randn(batch_size) #y
    a[:,:,3] = np.random.randn(batch_size).reshape(-1,1)*0.1 #y_vel
    
    for f in range(1,frames):
        a[:,f,[0,2]] = np.add(a[:,f-1,[0,2]], a[:,f,[1,3]] )
    
    yield a


# In[3]:


# for i in fmove_generator():
#     print(i[0])
#     break


# ## Bounce

# In[4]:


def bounce_generator(batch_size = 64):
    """
    Bounce generator. Bounce will happen either at x or y for whole batch.
    Yields mirrored x, x_vel, y, y_vel
    """
    a = np.zeros((batch_size, 2, 4))
    
    #randomly select whether bounce will be at x or y
    k = np.random.choice([0,2])
    l = 2-k
    
    #no bounce
    i = np.arange(batch_size)
    a[:,0,l] = np.random.uniform(size=batch_size)#.reshape(-1,1)
    a[:,0,l+1] = np.random.uniform(size=batch_size)*0.1#.reshape(-1,1)*0.1 #x_vel
    a[:,1,[l,l+1]] = a[:,0,[l,l+1]] #hard to broadcast on prev line
    
    #bounce
    border = np.random.choice([0,1], size=batch_size)
    vel = np.sign(border-0.5) * np.random.uniform(0, 0.1, size=batch_size)

    coord = border + np.random.uniform(size=batch_size) * vel

    a[:,0,k] = coord #.reshape(-1,1)
    a[:,1,k] = 2*border - coord
    a[:,0,k+1] = vel
    a[:,1,k+1] = -vel
    
    yield a   


# In[5]:


# for i in bounce_generator():
#     print(i.shape)
#     print(i)
#     break


# ## Attention

# In[6]:


def attention_generator(batch_size = 64):
    """
    Attention data generator. Yields x, x_vel, is_bounce (bool).
    """
    a = np.zeros((batch_size, 3))
    a[:,0] = np.random.choice([0,1], size=batch_size)          + np.random.choice([-1,1], size=batch_size)         * np.random.uniform(0, 0.1, size=batch_size)
    a[:,1] = np.random.uniform(size=batch_size)
    a[:,2] = ((a[:,0] > 1) + (a[:,0] < 0)) > 0 #clumsy 'or' statement
    yield a


# In[7]:


# for i in attention_generator():
#     print(i.shape)
#     print(i[0])
#     break


# ## Movement with bounce

# In[8]:


def bmove_generator(batch_size = 64, frames = 5, bounceAt = None):
    """
    Movement with bounces generator. Yields x, x_vel, y, y_vel with given number of frames.
    bounceAt=None will create the bounce at some random frame. Otherwise the bounce happens
    after the n-th frame (bounceAt = 3 means that the 4th frame is after the bounce)
    """
    while True:
        a = np.zeros((batch_size, frames, 4))
        #y
        y_vel = np.random.uniform(-0.05, 0.05, size=batch_size)
        a[:,:,2] = np.arange(frames)* y_vel.reshape(-1,1) + np.random.uniform(0.25,0.75) #no bounces here
        k = int(np.random.rand()*(frames-2)) if bounceAt == None else bounceAt
        a[:,:,3] = y_vel.reshape(-1,1)

        #x
        border = np.random.choice([0,1])
        x_vel = -np.random.uniform(0, 0.1, size=batch_size)

        if border == 1: x_vel = -x_vel
        delta = np.random.uniform(0, np.abs(x_vel))
        delta2 = np.abs(x_vel) - delta#np.abs(x_vel - delta)
        if border == 1: 
            delta = 1-delta
            delta2 = 1-delta2
        a[:,:k+1,1] = x_vel.reshape(-1,1)
        a[:,k+1:,1] = -x_vel.reshape(-1,1)

        for i in range(k+1):
            a[:,k-i,0] = delta - i * x_vel
        for i in range(frames-k-1):
            a[:,k+i+1,0] = delta2 - i * x_vel

        yield a

# ## Mirroring

def mirror_generator(batch_size = 64, frames = 5, bounceAt = None):
    """
    Mirroring generator. Yields x, x_vel, y, y_vel with given number of frames. Then Yields same but mirrored against border.
    bounceAt=None will create the bounce at some random frame. Otherwise the bounce happens
    after the n-th frame (bounceAt = 3 means that the 4th frame is after the bounce)
    """
    while True:
        a = np.zeros((batch_size, frames, 4))
        b = np.zeros((batch_size, frames, 4))
        #y
        y_vel = np.random.uniform(-0.05, 0.05, size=batch_size)
        a[:,:,2] = np.arange(frames)* y_vel.reshape(-1,1) + np.random.uniform(0.25,0.75) #no bounces here
        k = int(np.random.rand()*(frames-2)) if bounceAt == None else bounceAt
        a[:,:,3] = y_vel.reshape(-1,1)

        #x
        border = np.random.choice([0,1])
        x_vel = -np.random.uniform(0, 0.1, size=batch_size)

        if border == 1: x_vel = -x_vel
        delta = np.random.uniform(0, np.abs(x_vel))
        delta2 = np.abs(x_vel) - delta#np.abs(x_vel - delta)
        if border == 1: 
            delta = 1-delta
            delta2 = 1-delta2
        a[:,:k+1,1] = x_vel.reshape(-1,1)
        
        a[:,k+1:,1] = -x_vel.reshape(-1,1)

        for i in range(k+1):
            a[:,k-i,0] = delta - i * x_vel
        b = np.copy(a)
        for i in range(frames-k-1):
            a[:,k+i+1,0] = delta2 - i * x_vel
            b[:,k+i+1,0] = delta + (i+1) *x_vel

        yield b,a 
        

# In[9]:


# k = 0 
# for a in bmove_generator(frames = 5):
#     print(a[0])
#     k+=1
#     if k==1:
#         break


# ## Bounce movement with 2 ball collision
# Assume same mass for balls and 1 isn't moving.

# In[97]:


def R(gamma):
    """
    Returns rotation matrix to rotate a vector by gamma radians
    in counter-clockwise direction.
    """
    return np.array([
        [np.cos(gamma), -np.sin(gamma)],
        [np.sin(gamma), np.cos(gamma)]
    ])


# In[277]:


def bounce2_move_generator(batch_size = 64, frames = 5, bounceAt = None):
    """
    Movement with 2 ball bounce generator. Yields s_x, s_velx, s_y, s_vely, 
    p_x, p_velx, p_y, p_vely with given number of frames.
    bounceAt=None will create the bounce at some random frame. Otherwise the bounce happens
    after the n-th frame (bounceAt = 3 means that the 4th frame is after the bounce)
    """
    while True:
        a = np.zeros((batch_size, frames, 8))
        
        #moment of collision
        x,y = np.random.uniform(0.1,0.9, size=(2, batch_size)) #first ball coords
        x_dist = np.random.uniform(-0.1, 0.1, size=batch_size) #2radius is max dist
        y_dist = np.random.choice([-1,1], size=batch_size) * np.sqrt(0.1**2 - x_dist**2)
        x2, y2 = x+x_dist, y+y_dist  #second ball coords
        #rotate the normal vector by random amount
        #but still assure that bounce will happen
        theta = np.random.uniform(-np.pi/2, np.pi/2, size=batch_size)
        n = np.array([x_dist,y_dist]).T #normal vectors
        v = np.zeros((batch_size, 2))
        reverse_v = np.zeros((batch_size, 2)) #same but 180-degree turn for back-tracking
        for i in range(batch_size): 
            v[i] = np.matmul(R(theta[i]), n[i])
            reverse_v[i] = np.matmul(R(np.pi), v[i])
        speeds = np.random.randn(batch_size).reshape(-1,1)
        v *= speeds * 0.1 #velocities are also random
        reverse_v *= speeds * 0.1 

        #before the bounce
        deltax = np.random.uniform(0, v[:,0]) #before the bounce ball hasnt touched the other yet
        deltay = np.random.uniform(0, v[:,1])
        #point s
        a[:,:bounceAt+1,1] = v[:,0].reshape(-1,1)
        a[:,:bounceAt+1,3] = v[:,1].reshape(-1,1)
        #point p
        a[:,:bounceAt+1,4] = x2.reshape(-1,1)
        a[:,:bounceAt+1,6] = y2.reshape(-1,1)
        for t in np.arange(bounceAt, -1, -1):
            #point s
            a[:,t,0] = x - deltax - reverse_v[:,0] * t
            a[:,t,2] = y - deltay - reverse_v[:,1] * t

        #moment after the collision
        tan = y_dist/x_dist
        alpha = np.arctan(n[:,1]/n[:,0])
        beta = np.arctan(v[:,1]/v[:,0])
        #         print("ETA ", beta.shape)
        gamma = alpha - beta
        #         u = np.cos(gamma) * np.matmul(R(gamma),v) #this is vector of ball p
        #         p = np.sin(gamma) * np.matmul(R(gamma-np.pi/2),v) #this is the vector of ball s
        u, p = np.zeros((batch_size, 2)), np.zeros((batch_size, 2))
        #         print(v.shape, ' on v shape')
        for i in range(batch_size):
        #             tmp = np.cos(gamma[i]) * np.matmul(R(gamma[i]), v[i])
            u[i] = np.cos(gamma[i]) * np.matmul(R(gamma[i]), v[i])
            p[i] = np.sin(gamma[i]) * np.matmul(R(gamma[i]-np.pi/2), v[i])

        #after the bounce
        deltax2p = 1 - (deltax/v[:,0])
        deltay2p = 1 - (deltay/v[:,1])
        #how much each coordinate of both balls have moved
        deltaxs = deltax2p * u[:,0]
        deltays = deltax2p * u[:,1]
        deltaxp = deltax2p * p[:,0]
        deltayp = deltax2p * p[:,1]
        #point s
        a[:,bounceAt+1:,1] = p[:,0].reshape(-1,1)
        a[:,bounceAt+1:,3] = p[:,1].reshape(-1,1)
        #point p
        a[:,bounceAt+1:,5] = u[:,0].reshape(-1,1)
        a[:,bounceAt+1:,7] = u[:,1].reshape(-1,1)
        for t in np.arange(bounceAt+1, frames):
            #point s
            a[:,t,0] = x + deltaxs + u[:,0] * t
            a[:,t,2] = y + deltays + u[:,1] * t
            #point p
            a[:,t,4] = x2 + deltaxp + p[:,0] * t
            a[:,t,6] = y2 + deltayp + p[:,1] * t
        yield a


# In[281]:


# import pandas as pd
# df = pd.DataFrame(a[0])
# df.round(decimals=4)

# x, x_vel, y, y_vel, x2, x2_vel, y2, y2_vel = a[0,3]
# fig = plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# plt.xlim([0,1])
# plt.ylim([0,1])
# plt.scatter([x,x2],[y,y2], s=650)
# plt.quiver(x, y, x_vel, y_vel)
# #this (x_vel**2 + y_vel**2)

# x, x_vel, y, y_vel, x2, x2_vel, y2, y2_vel = a[0,4]
# #equals (x_vel+x2_vel)**2 + (y_vel+y2_vel)**2
# plt.subplot(1,2,2)
# plt.xlim([0,1])
# plt.ylim([0,1])
# plt.scatter([x,x2],[y,y2], s=650)
# plt.quiver(x,y,x_vel,y_vel)
# plt.quiver(x2,y2,x2_vel,y2_vel)

# for a in bounce2_move_generator(batch_size=1, bounceAt=3):
#     print(a)
#     break

