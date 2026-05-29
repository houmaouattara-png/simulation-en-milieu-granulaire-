from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches
import operator
import math
import sys
import numpy as np

# you can get the matplotlib figure and axis with
# these two variables 


def vec(x,y):
    """a simple function that returns a 2D numpy array"""
    return np.array([x,y],dtype=float)


class simu:
    """a simple static class that configure a simulation"""
    current_iter_number = 0
    grain_list   = []
    bond_list    = []
    patch_list   = []
    xlim         = (0,100)
    ylim         = (0,100)    
    _init_plot   = False
    custom_title = False
    msg_content  = ""
    fig, ax = plt.subplots()
    t, dt = 0., 0.

    def print(*msg):
        """display a msg in the left bottom corner of the simulation"""
        simu.msg_content = ''.join(map(str, msg))
    
    def init(tot_iter_number, update_plot_each, loop_function): 
        simu.tot_iter_number  = tot_iter_number
        simu.update_plot_each = update_plot_each
        simu.loop_function    = loop_function
        simu.init_plot()

    def add_object_to_scene(obj):
        if issubclass(type(obj), mpatches.Patch):
            simu.ax.add_patch(obj)
        simu.patch_list.append(obj)
        
    def init_plot():
        if (simu._init_plot is False):
            simu._init_plot = True
            # init matplotlib figure
            plt.xlim(simu.xlim)
            plt.ylim(simu.ylim)
            
            plt.gca().set_aspect('equal', adjustable='box')
            simu.title = simu.ax.text(0.5,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                                      transform=simu.ax.transAxes, ha="center")
            simu.msg = simu.ax.text(0.01,0.01, "", transform=simu.ax.transAxes, ha="left")


            for grain in simu.grain_list:
                grain.patch = plt.Circle((grain.pos[0], grain.pos[1]), grain.radius, facecolor=grain.color, edgecolor="black")
                simu.patch_list.append(grain.patch)
                simu.ax.add_patch(grain.patch)

            for bond in simu.bond_list:
                bond.patch = ConnectionPatch((bond.gr1.pos[0], bond.gr1.pos[1]), (bond.gr2.pos[0], bond.gr2.pos[1]),
                                             coordsA="data", coordsB="data",axesA=simu.ax,axesB=simu.ax)
                simu.patch_list.append(bond.patch)
                simu.ax.add_patch(bond.patch)

            simu.patch_list.append(simu.title)
            simu.patch_list.append(simu.msg)
            

    def remove_object_from_scene(obj):
        if hasattr(obj, 'patch'):
            if (obj.patch in simu.patch_list):
                simu.patch_list.remove(obj.patch)
                obj.patch.remove()
                

            

class grain: 
    """grain class that represent a circular discrete element with a 
 - radius :  self.rad, 
 - position : self.pos 
 - velocity : self.vel
 - acceleration : self.acc 
 - force : self.force
 - mass : self.mass
"""
    def __init__(self, pos, radius, density, color="tab:blue"): 
        self.radius = float(radius)
        x,y = pos
        self.pos     = vec(x,y)
        self.mass    = density*math.pi*self.radius**2
        self.vel     = vec(0.,0.)
        self.acc     = vec(0.,0.)
        self.force   = vec(0.,0.)
        self.color   = color
        self.visible = True
        self.index  = len(simu.grain_list)
        self.attached_bond = []
        self.bonded_grain  = []
        simu.grain_list.append(self)
        self.initial_pos = vec(x,y)
        

    def add_bond(self, b, gr):
        self.attached_bond.append(b)
        self.bonded_grain.append(gr)

    def is_bonded_to(self, gr):
        return (gr in self.bonded_grain)

    def remove(self):
        while (len(self.attached_bond) > 0):
            self.attached_bond[0].remove()
        simu.grain_list.remove(self)
        simu.remove_object_from_scene(self)
        

class bond:
    """a simple class that represents an elastic bond between two grains """
    def __init__(self, gr1, gr2): 
        self.gr1     = gr1
        self.gr2     = gr2
        self.lo      = np.linalg.norm(gr2.pos - gr1.pos)
        self.index   = len(simu.bond_list)
        self.surface = math.pi*((gr1.radius+gr2.radius)/2.)**2
        gr1.add_bond(self, gr2)
        gr2.add_bond(self, gr1)
        
        simu.bond_list.append(self)

    def remove(self):
        self.gr1.attached_bond.remove(self)
        self.gr2.attached_bond.remove(self)
        simu.bond_list.remove(self)
        simu.remove_object_from_scene(self)
        

    def update(self, stiffness=5e5, restitution_coef=0.1):
        rel_pos   = self.gr2.pos - self.gr1.pos
        dist      = np.linalg.norm(rel_pos)
        delta     = self.lo - dist
        normal    = rel_pos/dist
        force     = delta * stiffness
        if (force/self.surface) < -5e3:
            self.remove()
            return
        force = normal * force
        self.gr1.force -= force
        self.gr2.force += force
        
        # manage damping factor
        M  = (self.gr1.mass*self.gr2.mass)/(self.gr1.mass+self.gr2.mass);
        K  = stiffness;
        C  = 2.*(1./math.sqrt(1. + math.pow(math.pi/math.log(restitution_coef), 2)))*math.sqrt(K*M)
        V  = (self.gr2.vel - self.gr1.vel) * normal
        force2     = C * V * normal
        self.gr1.force += force2
        self.gr2.force -= force2
        
        
def contact(gr1, gr2, stiffness=5e5, restitution_coef=0.1, exclude_bonded_grain = False):
    """a function that computes contact between two grains. 
If the contact is detected, repulsive force are computed.
The repuslive force take into account stiffness and damping factor"""
    if (exclude_bonded_grain) is True:
        if gr1.is_bonded_to(gr2):
            return
    rel_pos   = gr2.pos - gr1.pos
    dist      = np.linalg.norm(rel_pos)
    delta     = -dist + gr1.radius + gr2.radius
    if (delta > 0.):
        # compute normal force 
        normal     = rel_pos/dist
        force1     = normal * delta * stiffness
        gr1.force -= force1
        gr2.force += force1

        # manage damping factor
        M  = (gr1.mass*gr2.mass)/(gr1.mass+gr2.mass);
        K  = stiffness;
        C  = 2.*(1./math.sqrt(1. + math.pow(math.pi/math.log(restitution_coef), 2)))*math.sqrt(K*M)
        V  = (gr2.vel - gr1.vel) * normal
        force2     = C * V * normal
        gr1.force += force2
        gr2.force -= force2

def in_contact(gr1, gr2, expand_ratio=1.):
    """a function that returns True if gr1 and gr2 are in contact. It returns False otherwise"""
    rel_pos   = gr2.pos - gr1.pos
    dist      = np.linalg.norm(rel_pos)
    return (gr1.radius*expand_ratio + gr2.radius*expand_ratio > dist)
    

def wall_contact(gr, delta, normal, stiffness=5e5):
    """a simple function that computes contact between a grain and a wall"""
    force1     = normal * delta * stiffness
    gr.force += force1
    

def save_domain(filename):
    """save domain in file as xyzr format"""
    with open(filename, 'w') as file:
        for grain in simu.grain_list:
            line = "{}\t{}\t{}\n".format(grain.pos[0], grain.pos[1], grain.radius)
            file.write(line)
        for bond in simu.bond_list:
            line = "{}\t{}\n".format(bond.gr1.index, bond.gr2.index)
            file.write(line)            
    print ("saving '{}'".format(filename))

    
def load_domain(filename):
    """load domain form a xyzr file"""
    with open(filename, 'r') as file:
        for line in file:
            data = line.split()
            if (len(data) == 3):
                x = float(data[0])
                y = float(data[1])
                r = float(data[2])
                gr = grain((x,y), r, 1)
            elif (len(data) == 2):
                gr1 = simu.grain_list[int(data[0])]
                gr2 = simu.grain_list[int(data[1])]
                b = bond(gr1,gr2)
    print ("loading '{}'".format(filename))


def _animate(i):
    """a private function required by matplotlib for updating the diagram"""
    info = "computing iteration = {}/{}".format(simu.current_iter_number, simu.tot_iter_number)
    # print(info, ' '*10, end='\r') I disable it because it causes slowdown with idle
    
    for i in range(simu.update_plot_each):
        simu.loop_function()
        simu.current_iter_number += 1
        simu.t += simu.dt

    # if simu.custom_title == False:
    #     simu.title.set_text(info)
    #     simu.msg.set_text(simu.msg_content)
    # else:
    #     simu.title.set_text(simu.msg_content)
    
    for grain in simu.grain_list:
        grain.patch.center = (grain.pos[0], grain.pos[1])
        grain.patch.radius = grain.radius
        grain.patch.set_facecolor(grain.color)
        grain.patch.set_visible(grain.visible)
        
    for bond in simu.bond_list:
        bond.patch.xy1 = (bond.gr1.pos[0], bond.gr1.pos[1])
        bond.patch.xy2 = (bond.gr2.pos[0], bond.gr2.pos[1])

    return simu.patch_list


class lcm: 
    """the lcm static class is a fast method for contact detection. 
It implements the Linked Cell Method that uses a grid for excluding 
non wanted colliding pairs and increase the speed of collision detection."""

    k                = 1.004
    domain_dimension = vec(0.,0.)
    point_min        = vec( 1000. , 1000.)
    point_max        = vec(-1000. ,-1000.)
    radius_max       = 0.
    grain_list       = simu.grain_list

    def update_domain():
        """this method update the bounding box of the grid"""
        
        lcm.radius_max = -1.
        for gr in lcm.grain_list:
            if (gr.radius > lcm.radius_max): lcm.radius_max = gr.radius

        for gr in lcm.grain_list:
            if (gr.pos[0]  < lcm.point_min[0]) : lcm.point_min[0] = gr.pos[0];
            if (gr.pos[1]  < lcm.point_min[1]) : lcm.point_min[1] = gr.pos[1];
            if (gr.pos[0]  > lcm.point_max[0]) : lcm.point_max[0] = gr.pos[0];
            if (gr.pos[1]  > lcm.point_max[1]) : lcm.point_max[1] = gr.pos[1];
            if (gr.radius > lcm.radius_max ) : lcm.radius_max = gr.radius;
        lcm.domain_dimension = lcm.point_max - lcm.point_min
        

        
    def compute_colliding_pair(expand_ratio=1.):
        """this method returns a list of possible colliding pairs"""
        lcm.update_domain()
        lcm.radius_max *= expand_ratio
        
        alpha = 2*lcm.k*lcm.radius_max
        C = math.floor(lcm.domain_dimension[0]/alpha)+1
        R = math.floor(lcm.domain_dimension[1]/alpha)+1

        lcm.grid =  [[[] for i in range(R+2)] for j in range(C+2)]
        

        for gr in lcm.grain_list:
            c = math.floor(C*(gr.pos[0] - lcm.point_min[0]) / (C*alpha)) + 1
            r = math.floor(R*(gr.pos[1] - lcm.point_min[1]) / (R*alpha)) + 1
            lcm.grid[c][r].append(gr)

        pair_list =[]
        for c in range(1,C+1):
            for r in range(1,R+1):
                for dc in range(-1,2):
                    for gr1 in lcm.grid[c][r]:
                        for gr2 in lcm.grid[c+dc][r+1]:
                            pair_list.append((gr1,gr2))
                
                for gr1 in lcm.grid[c][r]:
                    for gr2 in lcm.grid[c+1][r]:
                        pair_list.append((gr1,gr2))
                
                for i,gr1 in enumerate(lcm.grid[c][r]):
                    for j in range(i+1, len(lcm.grid[c][r])):
                        pair_list.append((gr1, lcm.grid[c][r][j]))

        return pair_list



def run(*, tot_iter_number, update_plot_each, loop_fn, video_name="test.mp4"):
    """Run the calculation here!"""
    simu.init(tot_iter_number, update_plot_each, loop_fn)
    
    def frame_generator():
        for i in range(tot_iter_number):
            if i % update_plot_each == 0:
                if loop_fn():
                    print(f"Stopping early at iteration {i} because loop_fn returned True.")
                    break
                yield i
    
    animate = animation.FuncAnimation(simu.fig, _animate, frames=frame_generator(), interval=2, blit=True, repeat=False)
    
    if video_name is not None:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=24, metadata=dict(artist='(c) minidem'), bitrate=1800)
        animate.save(video_name, writer=writer, dpi=400)
        print("saving '" + video_name + "'")
    else:
        plt.show()

