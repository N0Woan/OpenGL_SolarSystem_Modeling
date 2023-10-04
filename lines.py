from libs.buffer import *
from libs import transform as T
from libs.shader import *
from libs.transform import *
from utils import *
import ctypes
import glfw
import math
import numpy as np


def newline(radius, sides):
    vertices, indices, color, texcoords = [], [], [], []
    for j in range(sides+1):
        phi = 2 * np.pi * j / sides
        x = radius * np.cos(phi)
        y = radius * np.sin(phi)
        
        vertices += [[x, y, 0]]
        color += [[1, 1, 1]]
        indices += [j]
    indices += [0]
        
            
    vertices = np.array(vertices, dtype=np.float32)
    color = np.array(color, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32) 
    texcoords = np.array(texcoords, dtype=np.float32)       
    return vertices, indices, color, texcoords
    
    
class Lines(object):
    def __init__(self, vert_shader, frag_shader):   
        self.vertices_mercucy, self.indices, self.colors, self.texcoords = newline(9, 100)     
        self.vertices_venus, self.indices, self.colors, self.texcoords = newline(13, 100)
        self.vertices_earth, self.indices, self.colors, self.texcoords = newline(17, 100)
        self.vertices_mars, self.indices, self.colors, self.texcoords = newline(21, 100)
        self.vertices_jupiter, self.indices, self.colors, self.texcoords = newline(26, 100)
        self.vertices_saturn, self.indices, self.colors, self.texcoords = newline(34, 100)
        self.vertices_uranus, self.indices, self.colors, self.texcoords = newline(39, 100)
        self.vertices_neptune, self.indices, self.colors, self.texcoords = newline(42, 100)
        
        self.normals = generate_normals(self.vertices_mercucy, self.indices)
        
        self.vao_mercucy = VAO()
        self.vao_venus = VAO()
        self.vao_earth = VAO()
        self.vao_mars = VAO()
        self.vao_jupiter = VAO()
        self.vao_saturn = VAO()
        self.vao_uranus= VAO()
        self.vao_neptune = VAO()
        
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)

    def setup(self):
        self.vao_mercucy.add_vbo(0, self.vertices_mercucy, ncomponents=3, stride=0, offset=None)
        self.vao_mercucy.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao_mercucy.add_vbo(2, self.colors, ncomponents=3, stride=0, offset=None)
        # self.vao_mercucy.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)
        self.vao_mercucy.add_ebo(self.indices)
        
        self.vao_venus.add_vbo(0, self.vertices_venus, ncomponents=3, stride=0, offset=None)
        self.vao_venus.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao_venus.add_vbo(2, self.colors, ncomponents=3, stride=0, offset=None)
        # self.vao_venus.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)
        self.vao_venus.add_ebo(self.indices)
        
        self.vao_earth.add_vbo(0, self.vertices_earth, ncomponents=3, stride=0, offset=None)
        self.vao_earth.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao_earth.add_vbo(2, self.colors, ncomponents=3, stride=0, offset=None)
        # self.vao_earth.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)
        self.vao_earth.add_ebo(self.indices)
        
        self.vao_mars.add_vbo(0, self.vertices_mars, ncomponents=3, stride=0, offset=None)
        self.vao_mars.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao_mars.add_vbo(2, self.colors, ncomponents=3, stride=0, offset=None)
        # self.vao_mars.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)
        self.vao_mars.add_ebo(self.indices)
        
        self.vao_jupiter.add_vbo(0, self.vertices_jupiter, ncomponents=3, stride=0, offset=None)
        self.vao_jupiter.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao_jupiter.add_vbo(2, self.colors, ncomponents=3, stride=0, offset=None)
        # self.vao_jupiter.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)
        self.vao_jupiter.add_ebo(self.indices)
        
        self.vao_saturn.add_vbo(0, self.vertices_saturn, ncomponents=3, stride=0, offset=None)
        self.vao_saturn.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao_saturn.add_vbo(2, self.colors, ncomponents=3, stride=0, offset=None)
        # self.vao_saturn.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)
        self.vao_saturn.add_ebo(self.indices)
        
        self.vao_uranus.add_vbo(0, self.vertices_uranus, ncomponents=3, stride=0, offset=None)
        self.vao_uranus.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao_uranus.add_vbo(2, self.colors, ncomponents=3, stride=0, offset=None)
        # self.vao_uranus.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)
        self.vao_uranus.add_ebo(self.indices)
        
        self.vao_neptune.add_vbo(0, self.vertices_neptune, ncomponents=3, stride=0, offset=None)
        self.vao_neptune.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao_neptune.add_vbo(2, self.colors, ncomponents=3, stride=0, offset=None)
        # self.vao_neptune.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)
        self.vao_neptune.add_ebo(self.indices)

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
        modelview = view
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        self.uma.upload_uniform_matrix4fv(view, 'normalMat', True)
        self.uma.upload_uniform_scalar1i(0, 'option')

        self.vao_mercucy.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        
        self.vao_venus.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        
        self.vao_earth.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        
        self.vao_mars.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        
        self.vao_jupiter.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        
        self.vao_saturn.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        
        self.vao_uranus.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        
        self.vao_neptune.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)

    def key_handler(self, key):
        if key == glfw.KEY_1:
            self.selected_texture = 1
        if key == glfw.KEY_2:
            self.selected_texture = 2