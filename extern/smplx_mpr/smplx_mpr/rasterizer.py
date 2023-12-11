import torch

try:
    from .cuda.rasterizer import \
        estimate_normals as estimate_normals_cuda  # noqa: E501
    from .cuda.rasterizer import \
        project_mesh as project_mesh_cuda  # noqa: E501
    has_rasterizer = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_rasterizer = False
    import traceback
    stack_str = ''
    for line in traceback.format_stack():
        if 'frozen' not in line:
            stack_str += line + '\n'
    import_exception = traceback.format_exc() + '\n'
    import_exception = stack_str + import_exception
    import_exception = import_exception + \
        'Please reinstall mpr_smplx_render to build mpr_renderer.'


def estimate_normals(vertices, faces, pinhole, vertices_filter=None):
    """Estimate the vertices normals with the specified faces and camera.

    Args:
        vertices (torch.tensor): Shape should be (num_verts, 3).
        faces (torch.tensor): The faces of the vertices.
        pinhole (object): The object of the camera.

    Returns:
        coords (torch.tensor): The estimated coordinates.
        normals (torch.tensor): The estimated normals.
    """
    if vertices_filter is None:
        assert torch.is_tensor(vertices)
        assert vertices.is_cuda
        assert len(vertices.shape) == 2
        n = vertices.shape[0]
        vertices_filter = torch.ones((n),
                                     dtype=torch.uint8,
                                     device=vertices.device)
    vertices = vertices.contiguous()
    vertices_ndc = pinhole.project_ndc(vertices)
    coords, normals = estimate_normals_cuda(vertices_ndc, faces, vertices,
                                            vertices_filter, pinhole.h,
                                            pinhole.w)
    return coords, normals


def project_mesh(vertices,
                 faces,
                 vertice_values,
                 pinhole,
                 vertices_filter=None):
    """Project mesh to the image plane with the specified faces and camera.

    Args:
        vertices (torch.tensor): Shape should be (num_verts, 3).
        faces (torch.tensor): The faces of the vertices.
        vertice_values (torch.tensor): The depth of the each vertex.
        pinhole (object): The object of the camera.

    Returns:
        torch.tensor: The projected mesh.
    """
    if vertices_filter is None:
        assert torch.is_tensor(vertices)
        assert vertices.is_cuda
        assert len(vertices.shape) == 2
        n = vertices.shape[0]
        vertices_filter = torch.ones((n),
                                     dtype=torch.uint8,
                                     device=vertices.device)
    vertices = vertices.contiguous()
    vertices_ndc = pinhole.project_ndc(vertices)
    return project_mesh_cuda(vertices_ndc, faces, vertice_values,
                             vertices_filter, pinhole.h, pinhole.w)
