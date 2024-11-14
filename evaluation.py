import bpy
import os
import imageio

obj_file = "out/noise_point/mesh/mesh.obj"
kd_texture_file = "out/noise_point/mesh/texture_kd.png"
ks_texture_file = "out/noise_point/mesh/texture_ks.png"
EnvironmentMap_file = "interior.hdr"

img_ks = imageio.imread(ks_texture_file)
roughness = img_ks[:,:,1]
metallic = img_ks[:,:,2]
roughness_path = "out/noise_point/mesh/roughness.png"
metallic_path = "out/noise_point/mesh/metallic.png"

imageio.imwrite(roughness_path,roughness)
imageio.imwrite(metallic_path,metallic)

bpy.ops.wm.read_factory_settings(use_empty=True)

# Import the OBJ file
bpy.ops.import_scene.obj(filepath=obj_file)

# Set up the environment map
bpy.context.scene.world = bpy.data.worlds.new(name="NewWorld")
world = bpy.context.scene.world
world.use_nodes = True

env_node_tree = world.node_tree
env_background_node = env_node_tree.nodes.new(type='ShaderNodeTexEnvironment')
env_background_node.image = bpy.data.images.load(EnvironmentMap_file)

background_output_node = env_node_tree.nodes['Background']

# Link the environment map to the background
env_node_tree.links.new(env_background_node.outputs['Color'], background_output_node.inputs['Color'])

# Set up materials for the imported object
for obj in bpy.context.selected_objects:
    if obj.type == 'MESH':
        # Create a new material
        mat = bpy.data.materials.new(name="Material_with_Textures")
        mat.use_nodes = True
        
        # Get the material node tree
        node_tree = mat.node_tree
        nodes = node_tree.nodes
        links = node_tree.links

        # Clear default nodes
        for node in nodes:
            nodes.remove(node)
        
        # Add Principled BSDF
        principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        

        # Add Material Output
        material_output = nodes.new(type='ShaderNodeOutputMaterial')
        links.new(principled_bsdf.outputs['BSDF'], material_output.inputs['Surface'])

        # Add Diffuse Texture (Kd)
        kd_texture_node = nodes.new(type='ShaderNodeTexImage')
        kd_texture_node.image = bpy.data.images.load(kd_texture_file)
        links.new(kd_texture_node.outputs['Color'], principled_bsdf.inputs['Base Color'])

        # Add Roughness Texture (Ks)
        roughness_node = nodes.new(type='ShaderNodeTexImage')
        roughness_node.image = bpy.data.images.load(roughness_path)
        links.new(roughness_node.outputs['Color'], principled_bsdf.inputs['Roughness'])
        
        # Add Metallic Texture (Ks)
        metallic_node = nodes.new(type='ShaderNodeTexImage')
        metallic_node.image = bpy.data.images.load(metallic_path)
        links.new(metallic_node.outputs['Color'], principled_bsdf.inputs['Metallic'])

        # Assign the material to the object
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

# Set up camera
camera_data = bpy.data.cameras.new(name='Camera')
camera_object = bpy.data.objects.new('Camera', camera_data)
bpy.context.collection.objects.link(camera_object)
bpy.context.scene.camera = camera_object
camera_object.location = (4.12,-0.16,2.58)
camera_object.rotation_euler = (1.04, 0, 1.53)

# Set render settings
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'GPU'  # Use GPU if available
bpy.context.scene.render.filepath = "out/noise_point/mesh/relighting.png"
bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512
bpy.context.scene.render.film_transparent = True

# Render the image
bpy.ops.render.render(write_still=True)

print("Render complete. Image saved as render_result.png")
