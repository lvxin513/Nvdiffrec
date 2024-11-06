import torch
import torch.nn.functional as F

def differentiable_rendering(vertices, normals,light_position,light_direction,light_angle,view_position,albedo,ambient,diffuse,specular):
    #  Args:
    #     vertices (torch.Tensor): The vertices of the mesh (N, 3).
    #     normals (torch.Tensor): The normals at each vertex (N, 3).
    #     light_position (torch.Tensor): Position of the spotlight (3).
    #     light_direction (torch.Tensor): Direction of the spotlight (3).
    #     light_angle (float): The cutoff angle for the spotlight in radians.
    #     view_position (torch.Tensor): Position of the viewer/camera (3).
    #     albedo (torch.Tensor): Color information (N, 3).
    #     ambient_intensity (float): Intensity of the ambient light.
    #     diffuse_intensity (float): Intensity of the diffuse light.
    #     specular_intensity (float): Intensity of the specular light.
    #     shininess (int): Shininess coefficient for specular lighting.

    # Returns:
    #     torch.Tensor: Shaded colors for each vertex (N, 3)

    normals = F.normalize(normals, p=2, dim=1)
    light_direction = F.normalize(light_direction, p=2, dim=0)

    # Calculate light vector and normalize
    light_vec = light_position - vertices  # (N, 3)
    light_vec = F.normalize(light_vec, p=2, dim=1)

    # Calculate the spotlight effect (angle between light direction and light vector)
    cos_theta = torch.sum(light_vec * light_direction, dim=1).clamp(min=0, max=1)  # (N,)
    spotlight_factor = (cos_theta > torch.cos(light_angle)).float() * cos_theta  # (N,)

    # Ambient Component
    ambient = ambient * albedo  # (N, 3)

    # Diffuse Component (Lambertian shading)
    diffuse = diffuse * albedo * torch.clamp(torch.sum(normals * light_vec, dim=1, keepdim=True), min=0)  # (N, 3)

    # Specular Component (Blinn-Phong shading)
    view_vec = F.normalize(view_position - vertices, p=2, dim=1)  # (N, 3)
    halfway_vec = F.normalize(light_vec + view_vec, p=2, dim=1)  # (N, 3)
    specular = specular * torch.pow(torch.clamp(torch.sum(normals * halfway_vec, dim=1, keepdim=True), min=0), shininess)  # (N, 1)
    specular = specular.expand(-1, 3)  # (N, 3)

    # Combine all components
    shaded_color = ambient + spotlight_factor.unsqueeze(1) * (diffuse + specular)  # (N, 3)
    shaded_color = torch.clamp(shaded_color, min=0, max=1)  # Ensure the colors are within a valid range

    return shaded_color

# Example usage
vertices = torch.rand((100, 3), requires_grad=True)  # Example vertices
normals = torch.rand((100, 3), requires_grad=True)  # Example normals
light_position = torch.tensor([10.0, 10.0, 10.0])  # Spotlight position
light_direction = torch.tensor([-1.0, -1.0, -1.0])  # Spotlight direction
light_angle = torch.tensor(0.5)  # Spotlight cutoff angle in radians
view_position = torch.tensor([0.0, 0.0, 10.0])  # Camera position
albedo = torch.rand((100, 3))  # Color per vertex

shaded_output = differentiable_rendering(vertices, normals, light_position, light_direction, light_angle, view_position, albedo)
print(shaded_output)