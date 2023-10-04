from libs.buffer import *
from libs import transform as T
from libs.shader import *
from libs.transform import *
from utils import *
from time import *
import ctypes
import glfw
import math
import numpy as np
import random



def newring(inner_radius, outer_radius, sides):
    vertices, indices, color, texcoords = [], [], [], []
    for j in range(sides+1):
        phi = 2 * np.pi * j / sides
        outer_x = outer_radius * np.cos(phi)
        outer_y = outer_radius * np.sin(phi)
        inner_x = inner_radius * np.cos(phi)
        inner_y = inner_radius * np.sin(phi)
        
        vertices += [[outer_x, outer_y, 0], [inner_x, inner_y, 0]]
        color += [[0, 0, 1], [0, 0, 1]]
        texcoords += [[j / sides, 1], [j / sides, 0]]
    
    for j in range(2*(sides+1)):
        indices += [j]
        
    vertices = np.array(vertices, dtype=np.float32)
    color = np.array(color, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32) 
    texcoords = np.array(texcoords, dtype=np.float32)       
    return vertices, indices, color, texcoords


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
            color += [[0, 0, 1]]
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



class Solar(object):
    def __init__(self, vert_shader, frag_shader):   
        #background
        self.vertices_background, self.indices, self.colors, self.texcoords = newsphere(200, 30)  
        self.vao_background = VAO()
        
        #sun
        self.vertices_sun, self.indices, self.colors, self.texcoords = newsphere(5, 30)  
        self.vao_sun = VAO()
        
        #earth
        self.vertices_earth, self.indices, self.colors, self.texcoords = newsphere(1, 30)
        self.vao_earth = VAO()
        self.orbit_earth = neworbit(17, 364*24*4)
        self.frame_earth = 0
        self.rot_earth = rotate(vec((1,0,0)), angle=23.44)
        
        #moon
        self.vertices_moon, self.indices, self.colors, self.texcoords = newsphere(0.2, 30)    
        self.vao_moon = VAO()
        self.orbit_moon = neworbit(2, 28*24*4)
        self.frame_moon = 0
        self.rot_moon = rotate(vec((1,0,0)), angle=0)
        
        #mercucy
        self.vertices_mercucy, self.indices, self.colors, self.texcoords = newsphere(0.38, 30)
        self.vao_mercucy = VAO()
        self.orbit_mercucy = neworbit(9, 364*24)
        self.frame_mercucy = random.randint(0, 364*24)
        self.rot_mercucy = rotate(vec((1,0,0)), angle=0)
        
        #venus
        self.vertices_venus, self.indices, self.colors, self.texcoords = newsphere(0.95, 30)
        self.vao_venus = VAO()
        self.orbit_venus = neworbit(13, int(364*24*4*225/365))
        self.frame_venus = random.randint(0, int(364*24*4*225/365))
        self.rot_venus = rotate(vec((1,0,0)), angle=117.3)
        
        #mars
        self.vertices_mars, self.indices, self.colors, self.texcoords = newsphere(0.53, 30)
        self.vao_mars = VAO()
        self.orbit_mars = neworbit(21, int(364*25*4*669/365))
        self.frame_mars = random.randint(0, int(364*25*4*669/365))
        self.rot_mars = rotate(vec((1,0,0)), angle=25.19)
        
        #jupiter
        self.vertices_jupiter, self.indices, self.colors, self.texcoords = newsphere(2, 30)
        self.vao_jupiter = VAO()
        self.orbit_jupiter = neworbit(26, 364*24*4*12)
        self.frame_jupiter = random.randint(0, 364*24*4*12)
        self.rot_jupiter = rotate(vec((1,0,0)), angle=0)
        
        #saturn
        self.vertices_saturn, self.indices, self.colors, self.texcoords = newsphere(1.68, 30)
        self.vao_saturn = VAO()
        self.orbit_saturn = neworbit(34, 364*24*4*30)
        self.frame_saturn = random.randint(0, 364*24*4*30)
        self.rot_saturn = rotate(vec((1,0,0)), angle=26.7)
        
        #saturn_ring
        self.vertices_saturn_ring, self.indices_saturn_ring, self.colors_saturn_ring, self.texcoords_saturn_ring = newring(2, 3, 30)
        self.vao_saturn_ring = VAO()
        self.orbit_saturn_ring = neworbit(34, 364*24*4*30)
        self.frame_saturn_ring = self.frame_saturn
        self.rot_saturn_ring = rotate(vec((1,0,0)), angle=26.7)
        self.normals_saturn_ring = generate_normals(self.vertices_saturn_ring, self.indices_saturn_ring)
        
        #uranus
        self.vertices_uranus, self.indices, self.colors, self.texcoords = newsphere(1.5, 30)
        self.vao_uranus = VAO()
        self.orbit_uranus = neworbit(39, 364*24*4*84)
        self.frame_uranus = random.randint(0, 364*24*4*84)
        self.rot_uranus = rotate(vec((1,0,0)), angle=98)
        
        #neptune
        self.vertices_neptune, self.indices, self.colors, self.texcoords = newsphere(1.3, 30)
        self.vao_neptune = VAO()
        self.orbit_neptune = neworbit(42, 364*24*4*165)
        self.frame_neptune = random.randint(0, 364*24*4*165)
        self.rot_neptune = rotate(vec((1,0,0)), angle=30)
        
        #general
        self.normals = generate_normals(self.vertices_sun, self.indices)
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)

    def setup(self):
        #background
        self.vao_background.add_vbo(0, self.vertices_background, ncomponents=3, stride=0, offset=None)
        self.vao_background.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        # self.vao_background.add_vbo(2, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao_background.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)
        self.vao_background.add_ebo(self.indices)
        
        #sun
        self.vao_sun.add_vbo(0, self.vertices_sun, ncomponents=3, stride=0, offset=None)
        self.vao_sun.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        # self.vao_sun.add_vbo(2, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao_sun.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)
        self.vao_sun.add_ebo(self.indices)
        
        #earth
        self.vao_earth.add_vbo(0, self.vertices_earth, ncomponents=3, stride=0, offset=None)
        self.vao_earth.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        # self.vao_earth.add_vbo(2, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao_earth.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)
        self.vao_earth.add_ebo(self.indices)
        
        #moon
        self.vao_moon.add_vbo(0, self.vertices_moon, ncomponents=3, stride=0, offset=None)
        self.vao_moon.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        # self.vao_moon.add_vbo(2, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao_moon.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)
        self.vao_moon.add_ebo(self.indices)

        #mercucy
        self.vao_mercucy.add_vbo(0, self.vertices_mercucy, ncomponents=3, stride=0, offset=None)
        self.vao_mercucy.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        # self.vao_mercucy.add_vbo(2, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao_mercucy.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)
        self.vao_mercucy.add_ebo(self.indices)
        
        #venus
        self.vao_venus.add_vbo(0, self.vertices_venus, ncomponents=3, stride=0, offset=None)
        self.vao_venus.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        # self.vao_venus.add_vbo(2, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao_venus.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)
        self.vao_venus.add_ebo(self.indices)
        
        #mars
        self.vao_mars.add_vbo(0, self.vertices_mars, ncomponents=3, stride=0, offset=None)
        self.vao_mars.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        # self.vao_mars.add_vbo(2, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao_mars.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)
        self.vao_mars.add_ebo(self.indices)
        
        #jupiter
        self.vao_jupiter.add_vbo(0, self.vertices_jupiter, ncomponents=3, stride=0, offset=None)
        self.vao_jupiter.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        # self.vao_jupiter.add_vbo(2, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao_jupiter.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)
        self.vao_jupiter.add_ebo(self.indices)
        
        #saturn
        self.vao_saturn.add_vbo(0, self.vertices_saturn, ncomponents=3, stride=0, offset=None)
        self.vao_saturn.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        # self.vao_saturn.add_vbo(2, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao_saturn.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)
        self.vao_saturn.add_ebo(self.indices)
        
        #saturn ring
        self.vao_saturn_ring.add_vbo(0, self.vertices_saturn_ring, ncomponents=3, stride=0, offset=None)
        self.vao_saturn_ring.add_vbo(1, self.normals_saturn_ring, ncomponents=3, stride=0, offset=None)
        # self.vao_saturn_ring.add_vbo(2, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao_saturn_ring.add_vbo(3, self.texcoords_saturn_ring, ncomponents=2, stride=0, offset=None)
        self.vao_saturn_ring.add_ebo(self.indices_saturn_ring)
        
        #uranus
        self.vao_uranus.add_vbo(0, self.vertices_uranus, ncomponents=3, stride=0, offset=None)
        self.vao_uranus.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        # self.vao_uranus.add_vbo(2, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao_uranus.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)
        self.vao_uranus.add_ebo(self.indices)
        
        #neptune
        self.vao_neptune.add_vbo(0, self.vertices_neptune, ncomponents=3, stride=0, offset=None)
        self.vao_neptune.add_vbo(1, self.normals, ncomponents=3, stride=0, offset=None)
        # self.vao_neptune.add_vbo(2, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao_neptune.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)
        self.vao_neptune.add_ebo(self.indices)
        
        #general setup
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
        
        self.uma.setup_texture('texture0', 'D:/Workplace/Source Code/Python/CG/Solar/resources/textures/sun.jpg')
        self.uma.setup_texture('texture1', 'D:/Workplace/Source Code/Python/CG/Solar/resources/textures/earth.jpg')
        self.uma.setup_texture('texture2', 'D:/Workplace/Source Code/Python/CG/Solar/resources/textures/moon.jpeg')
        self.uma.setup_texture('texture3', 'D:/Workplace/Source Code/Python/CG/Solar/resources/textures/mercucy.jpg')
        self.uma.setup_texture('texture4', 'D:/Workplace/Source Code/Python/CG/Solar/resources/textures/venus.jpg')
        self.uma.setup_texture('texture5', 'D:/Workplace/Source Code/Python/CG/Solar/resources/textures/mars.jpg')
        self.uma.setup_texture('texture6', 'D:/Workplace/Source Code/Python/CG/Solar/resources/textures/jupiter.jpg')
        self.uma.setup_texture('texture7', 'D:/Workplace/Source Code/Python/CG/Solar/resources/textures/saturn.jpg')
        self.uma.setup_texture('texture8', 'D:/Workplace/Source Code/Python/CG/Solar/resources/textures/ring.jpg')
        self.uma.setup_texture('texture9', 'D:/Workplace/Source Code/Python/CG/Solar/resources/textures/uranus.jpg')
        self.uma.setup_texture('texture10', 'D:/Workplace/Source Code/Python/CG/Solar/resources/textures/neptune.jpg')
        self.uma.setup_texture('texture11', 'D:/Workplace/Source Code/Python/CG/Solar/resources/textures/background.jpg')
        
        return self

    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(view, 'normalMat', True)
        
        #background
        modelview = view
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        self.uma.upload_uniform_scalar1i(11, 'option')
        self.vao_background.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        
        #sun
        modelview = view
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        self.uma.upload_uniform_scalar1i(0, 'option')
        self.vao_sun.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        
        #earth
        self.frame_earth %= len(self.orbit_earth)
        trans = translate(vec((self.orbit_earth[self.frame_earth][0], self.orbit_earth[self.frame_earth][1], self.orbit_earth[self.frame_earth][2])))
        self.rot_earth = rotate(vec((0,-0.3971479,0.91775465)), angle=360/96) @ self.rot_earth
        modelview = view @ trans @ self.rot_earth
        self.frame_earth += 1
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        self.uma.upload_uniform_scalar1i(1, 'option')
        self.vao_earth.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        
        #moon
        self.frame_moon %= len(self.orbit_earth)
        frame_moon = int((self.frame_moon*13)%len(self.orbit_moon))
        trans = translate(vec((self.orbit_moon[frame_moon][0]+self.orbit_earth[self.frame_moon][0], self.orbit_moon[frame_moon][1]+self.orbit_earth[self.frame_moon][1], self.orbit_moon[frame_moon][2]+self.orbit_earth[self.frame_moon][2])))
        self.rot_moon = rotate(vec((0,0,1)), angle=360/(96*28)) @ self.rot_moon
        modelview = view @ trans @ self.rot_moon
        self.frame_moon += 1
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        self.uma.upload_uniform_scalar1i(2, 'option')
        self.vao_moon.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        
        #mercucy
        self.frame_mercucy %= len(self.orbit_mercucy)
        trans = translate(vec((self.orbit_mercucy[self.frame_mercucy][0], self.orbit_mercucy[self.frame_mercucy][1], self.orbit_mercucy[self.frame_mercucy][2])))
        self.rot_mercucy = rotate(vec((0,0,1)), angle=360/(96*59)) @ self.rot_mercucy
        modelview = view @ trans @ self.rot_mercucy
        self.frame_mercucy += 1
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        self.uma.upload_uniform_scalar1i(3, 'option')
        self.vao_mercucy.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        
        #venus
        self.frame_venus %= len(self.orbit_venus)
        trans = translate(vec((self.orbit_venus[self.frame_venus][0], self.orbit_venus[self.frame_venus][1], self.orbit_venus[self.frame_venus][2])))
        self.rot_venus = rotate(vec((0,-0.8886172,-0.45864955)), angle=360/(96*241)) @ self.rot_venus
        modelview = view @ trans @ self.rot_venus
        self.frame_venus += 1
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        self.uma.upload_uniform_scalar1i(4, 'option')
        self.vao_venus.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        
        #mars
        self.frame_mars %= len(self.orbit_mars)
        trans = translate(vec((self.orbit_mars[self.frame_mars][0], self.orbit_mars[self.frame_mars][1], self.orbit_mars[self.frame_mars][2])))
        self.rot_mars = rotate(vec((0,-0.42562136,0.9049013)), angle=360/(96*25/24)) @ self.rot_mars
        modelview = view @ trans @ self.rot_mars
        self.frame_mars += 1
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        self.uma.upload_uniform_scalar1i(5, 'option')
        self.vao_mars.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        
        #jupiter
        self.frame_jupiter %= len(self.orbit_jupiter)
        trans = translate(vec((self.orbit_jupiter[self.frame_jupiter][0], self.orbit_jupiter[self.frame_jupiter][1], self.orbit_jupiter[self.frame_jupiter][2])))
        self.rot_jupiter = rotate(vec((0,0,1)), angle=360/(96*9/24)) @ self.rot_jupiter
        modelview = view @ trans @ self.rot_jupiter
        self.frame_jupiter += 1
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        self.uma.upload_uniform_scalar1i(6, 'option')
        self.vao_jupiter.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        
        #saturn
        self.frame_saturn %= len(self.orbit_saturn)
        trans = translate(vec((self.orbit_saturn[self.frame_saturn][0], self.orbit_saturn[self.frame_saturn][1], self.orbit_saturn[self.frame_saturn][2])))
        self.rot_saturn = rotate(vec((0,-0.449319,0.8933714)), angle=360/(96*11/24)) @ self.rot_saturn
        modelview = view @ trans @ self.rot_saturn
        self.frame_saturn += 1
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        self.uma.upload_uniform_scalar1i(7, 'option')
        self.vao_saturn.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        
        #saturn ring
        self.frame_saturn_ring %= len(self.orbit_saturn_ring)
        trans = translate(vec((self.orbit_saturn_ring[self.frame_saturn_ring][0], self.orbit_saturn_ring[self.frame_saturn_ring][1], self.orbit_saturn_ring[self.frame_saturn_ring][2])))
        modelview = view @ trans @ self.rot_saturn_ring
        self.frame_saturn_ring += 1
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        self.uma.upload_uniform_scalar1i(8, 'option')
        self.vao_saturn_ring.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        
        #uranus
        self.frame_uranus %= len(self.orbit_uranus)
        trans = translate(vec((self.orbit_uranus[self.frame_uranus][0], self.orbit_uranus[self.frame_uranus][1], self.orbit_uranus[self.frame_uranus][2])))
        self.rot_uranus = rotate(vec((0,-0.99026805,-0.1391731)), angle=360/(96*18/24)) @ self.rot_uranus
        modelview = view @ trans @ self.rot_uranus
        self.frame_uranus += 1
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        self.uma.upload_uniform_scalar1i(9, 'option')
        self.vao_uranus.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        
        #neptune
        self.frame_neptune %= len(self.orbit_neptune)
        trans = translate(vec((self.orbit_neptune[self.frame_neptune][0], self.orbit_neptune[self.frame_neptune][1], self.orbit_neptune[self.frame_neptune][2])))
        self.rot_neptune = rotate(vec((0,-0.5,0.8660254)), angle=360/(96*18/24)) @ self.rot_neptune
        modelview = view @ trans @ self.rot_neptune
        self.frame_neptune += 1
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        self.uma.upload_uniform_scalar1i(10, 'option')
        self.vao_neptune.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP,
                          self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        
        sleep(0.0001)
        

    def key_handler(self, key):
        if key == glfw.KEY_1:
            self.selected_texture = 1
        if key == glfw.KEY_2:
            self.selected_texture = 2  