bl_info = {
    "name": "Import Planar Code",
    "author": "Naoya Tsuruta",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "File > Import > Planar Code",
    "description": "Import planar code and construct mesh by assigning vertex positions.",
    "warning": "",
    "support": "TESTING",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Import-Export",
}

import bpy
import bmesh
import numpy as np
import mathutils as mu
from bpy.props import StringProperty, IntProperty, BoolProperty
import struct
import collections
import os
import random

class PlanarCodeReader:
    def __init__(self, filename, index, embed2d, embed3d):
        self.faceCounters = []
        verts_loc, faces = self.read(filename, index)
        if (not verts_loc):
            return
        if (len(verts_loc) <= 0):
            return
        # create new mesh
        name = os.path.basename(filename) + "_" + str(index)
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(verts_loc,[],faces)
        mesh.update(calc_edges=True)
        
        # create new bmesh
        bm = bmesh.new()
        bm.from_mesh(mesh)
        # enable lookup
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        if (embed2d):
            pv = self.embed(bm)
            print(pv)
            if (embed3d):
                self.liftup(bm, pv)
        bm.to_mesh(mesh)

        # create new object
        obj = bpy.data.objects.new(name, mesh)
        # set object location
        obj.location = bpy.context.scene.cursor.location
        # link the object to collection
        bpy.context.scene.collection.objects.link(obj)

    def read(self, filename, index):
        self.f = open(filename, "rb")
        verts = []
        faces = []   
        try:
            DEFAULT_HEADER = b">>planar_code<<"
            header = self.f.read(len(DEFAULT_HEADER))
            if (header == DEFAULT_HEADER):
                print(index)
                self.skip(index)
                # create verts
                num_vert = struct.unpack('b', self.f.read(1))
                i = 0
                while i < num_vert[0]:
                    # create vertex
                    verts.append((0, 0, 0))
                    # read adjacant vertices
                    adj = []
                    while True: 
                        tmp = struct.unpack('b', self.f.read(1))
                        if (tmp[0] <= 0): # 0 means separator
                            break
                        adj.append(tmp[0])
                    # add face counter 
                    lastIndex = len(adj)-1
                    for j in range(lastIndex):
                        self.addIfAbsent(collections.Counter([i, adj[j+1]-1, adj[j]-1]))
                    self.addIfAbsent(collections.Counter([i, adj[0]-1, adj[lastIndex]-1]))
                    i += 1
                for counter in self.faceCounters:
                    faces.append(tuple(counter))
        except:
            print(f"Error in reading {filename}")
            self.f.close()
            return
        self.f.close()
        del self.f
        return verts, faces
    
    def skip(self, index):
        # skip to target index
        for i in range(index):
            num_vert = struct.unpack('b', self.f.read(1))
            n = num_vert[0]
            while n > 0:
                d = struct.unpack('b', self.f.read(1))
                if (d[0] == 0):
                    n -= 1

    def addIfAbsent(self, fc):
        for counter in self.faceCounters:
            if (counter == fc):
                break
        else:
            self.faceCounters.append(fc)
    
    def embed(self, bm):
        # randomly pick up a face
        outerFace = bm.faces[random.randint(0, len(bm.faces)-1)]
        # embed an outer face to form a regular polygon inscribed into a circle
        n = len(outerFace.verts)
        inv_sqrt = 1.0 / np.sqrt(n)
        angle = 360.0 / n
        for i, v in enumerate(outerFace.verts):
            rad = (-i * angle / 180.0) * np.pi
            x = inv_sqrt * np.cos(rad)
            y = inv_sqrt * np.sin(rad)
            v.co.x = x
            v.co.y = y
        rests = []
        for v in bm.verts:
            if (not v in outerFace.verts):
                rests.append(v)
        # variables for the force F_uv on a Edge(u,v)
        fuv = np.zeros((len(bm.edges), 3))
        # force F_v on a Vertex(v)
        fv = np.zeros((len(bm.verts), 3))
        # Constant value
        n_pi = np.sqrt(len(bm.verts) / np.pi)
        # final double A = 2.5;
        avg_area = np.pi / len(bm.verts)
        loop = 0

        # iterations
        while (loop < 500):
            # Set F_v to zero
            fv[:] = 0

            # Calculate F_uv for Edges
            for j, e in enumerate(bm.edges):
                v = e.verts[0]
                u = e.verts[1]
                C = n_pi
                x = C * np.power(v.co.x - u.co.x, 3)
                y = C * np.power(v.co.y - u.co.y, 3)
                if (np.isfinite(x) and np.isfinite(y)):
                    fuv[j] = [x, y, 0]
                # Update the forces on v and u
                fv[v.index] -= fuv[j]
                fv[u.index] += fuv[j]
			
            # Move Vertices
            cool = np.sqrt(avg_area) / (1.0 + np.power(np.sqrt(avg_area * loop), 3))
            for v in rests:
                f = np.linalg.norm(fv[v.index])
                size = min(f, cool)
                if f != 0:
                    fv[v.index] /= f
                fv[v.index] *= size
                v.co.x += fv[v.index, 0]
                v.co.y += fv[v.index, 1]
            loop += 1
        return self.periphericity(bm, outerFace)

    def periphericity(self, bm, outer):
        stack0 = []
        stack1 = []
        per = 0

        pv = np.full(len(bm.verts), -1)
        for v in outer.verts:
            stack0.append(v)

        while (stack0):
            for v in stack0:
                pv[v.index] = per
            
            # Search adjoining verts
            for vi in stack0:
                links = vi.link_edges
                for e in links:
                    vo = e.verts[1] if vi.index == e.verts[0].index else e.verts[0]
                    if (pv[vo.index] < 0):
                        if (not vo in stack1):
                            stack1.append(vo)
            stack0.clear()
            stack0.extend(stack1)
            stack1.clear()
            per += 1
        return pv
    
    def liftup(self, bm, pv):
        H = 0.3
        for v in bm.verts:
            z = H * pv[v.index]
            v.co.z = z
	

    
class IMPORT_OT_pcode(bpy.types.Operator):
    """Import Planar Code Operator"""

    bl_idname = "import_planarcode.pcode"
    bl_label = "Import planar code"
    bl_description = "Embed a graph written in planar code (binary file)"
    bl_options = {"REGISTER", "UNDO"}

    bpy.types.Scene.ch = None
    bpy.types.Scene.poly = None

    filepath: StringProperty(
        name="File Path",
        description="Filepath used for importing the Planar code file",
        maxlen=1024,
        default="",
    )
    CODE_INDEX: IntProperty(
        name="Planer code index",
        description="An index follows generated order",
        default=0,
        min=0,
    )
    EMBED_2D: BoolProperty(
        name="Embedding in 2D",
        description="Embed a graph in the plane",
        default=True,
    )
    EMBED_3D: BoolProperty(
        name="Realizing a graph",
        description="Make a polyhedron by giving the heights to the vertices",
        default=True,
    )

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        PlanarCodeReader(self.filepath, self.CODE_INDEX, self.EMBED_2D, self.EMBED_3D)
        return {"FINISHED"}

def menu_func(self, context):
    self.layout.operator(IMPORT_OT_pcode.bl_idname, text="Planar code (.*)")

classes = (
    IMPORT_OT_pcode,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.TOPBAR_MT_file_import.append(menu_func)

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func)

if __name__ == "__main__":
    register()

