from libs.buffer import *
from libs import transform as T
from libs.shader import *
from libs.transform import *
from utils import *
import ctypes
import glfw
import math
import numpy as np


def newsphere(radius, sides):
    vertices, indices, color, texcoords = [], [], [], []
    for i in range(sides+1):
        for j in range(sides+1):
            theta = np.pi * i / sides
            phi = 2 * np.pi * j / sides
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            
            vertices += [[x, y, z]]
            color += [[255, 255, 255]]
            texcoords += [[j / sides, i / sides]]
            
    
    for j in range(sides):
        for i in range(sides):
            point = (sides+1)*j+i
            indices += [point, point+sides+1, point+1, point+sides+2]
            
    vertices = np.array(vertices, dtype=np.float32)
    color = np.array(color, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32) 
    texcoords = np.array(texcoords, dtype=np.float32)       
    return vertices, indices, color, texcoords


def neworbit(radius, sides):
    orbit = []
    for i in range(sides):
        phi = 2 * np.pi * i / sides
        x = radius * np.cos(phi)
        y = radius * np.sin(phi)
        orbit += [[x, y, 0]]
    return np.array(orbit, dtype=np.float32)
    
    
class Moon(object):
    def __init__(self, radius, sides, orbit_radius, orbit_sides, earth_orbit, vert_shader, frag_shader):   
        self.vertices, self.indices, self.colors, self.texcoords = newsphere(radius, sides)     
        self.normals = generate_normals(self.vertices, self.indices)

        self.earth_orbit = earth_orbit
        self.orbit = neworbit(orbit_radius, orbit_sides)
        self.frame = 0
        self.rot = rotate(vec((1,0,0)), angle=23.4)
        
        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)
        
        self.vao.add_ebo(self.indices)

        normalMat = np.identity(4, 'f')
        projection = T.ortho(-0.5, 2.5, -0.5, 1.5, -1, 1)
        modelview = np.identity(4, 'f')

        # Light
        I_light = np.array([
            [0.5, 0.5, 0.5],  # diffuse
            [1.0, 1.0, 1.0],  # specular
            [0.5, 0.5, 0.5]  # ambient
        ], dtype=np.float32).T
        light_pos = np.array([0, 0, -1000], dtype=np.float32)

        # Materials
        K_materials = np.array([
            [0.54,      0.89,       0.63],  # diffuse
            [0.316228,	0.316228,	0.316228],  # specular
            [0.135,	    0.2225,	    0.1575]  # ambient
        ], dtype=np.float32).T
        
        shininess = 200.0
        mode = 2

        GL.glUseProgram(self.shader.render_idx)
        self.uma.upload_uniform_matrix4fv(normalMat, 'normalMat', True)
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
        self.uma.upload_uniform_vector3fv(light_pos, 'light_pos')

        self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)
        self.uma.upload_uniform_scalar1f(shininess, 'shininess')
        self.uma.upload_uniform_scalar1i(mode, 'mode')
        
        return self

    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)
        self.frame %= len(self.earth_orbit)
        frame = int((self.frame*13)%len(self.orbit))
        trans = translate(vec((self.orbit[frame][0]+self.earth_orbit[self.frame][0], self.orbit[frame][1]+self.earth_orbit[self.frame][1], self.orbit[frame][2]+self.earth_orbit[self.frame][2])))
        self.rot = rotate(vec((0,-0.3971479,0.91775465)), angle=360/96) @ self.rot
        modelview = view @ trans 
        self.frame += 1

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        self.uma.upload_uniform_matrix4fv(view, 'normalMat', True)
        self.uma.upload_uniform_scalar1i(0, 'option')

        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)

    def key_handler(self, key):
        if key == glfw.KEY_1:
            self.selected_texture = 1
        if key == glfw.KEY_2:
            self.selected_texture = 2