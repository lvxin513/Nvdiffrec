import torch
import torch.nn.functional as F

def differentiable_rendering(vertices, normals,light_position,light_intensity,view_position,albedo,roughness):

    normals = F.normalize(normals, p=2, dim=-1)
    light_direction = light_position - vertices
    light_direction = F.normalize(light_direction, p=2, dim=-1)

    # Diffuse Component (Lambertian shading)
    
    dif = torch.sum(normals * light_direction, dim=-1, keepdim=True)
    diffuse = light_intensity * albedo * torch.clamp(dif, min=0)  # (N, 3)

    # Specular Component (Blinn-Phong shading)
    shininess = 1.0 / (roughness * roughness)
    # print(shininess)
    view_vec = F.normalize(view_position - vertices, p=2, dim=-1)  # (N, 3)
    halfway_vec = F.normalize(light_direction + view_vec, p=2, dim=-1)  # (N, 3)
    specular = light_intensity * torch.pow(torch.clamp(torch.sum(normals * halfway_vec, dim=-1, keepdim=True), min=0), shininess)
    
    # specular = specular.expand(-1, 3)  # (N, 3)

    # Combine all components
    shaded_color = diffuse + specular  # (N, 3)
    shaded_color = torch.clamp(shaded_color, min=0, max=1)  # Ensure the colors are within a valid range

    return shaded_color
