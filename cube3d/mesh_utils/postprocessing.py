import logging

import numpy as np

try:
    import pymeshlab  # ty: ignore[unresolved-import]  # optional

    PYMESHLAB_AVAILABLE = True
except ImportError:
    PYMESHLAB_AVAILABLE = False
    from types import SimpleNamespace
    from typing import Any

    # Stub for typing/attribute-access when pymeshlab isn't installed.
    # All attributes return Any so calls and annotations type-check.
    class _PymeshlabStub(SimpleNamespace):
        def __getattr__(self, _name: str) -> Any:
            return Any

    pymeshlab = _PymeshlabStub()


def create_pymeshset(vertices: np.ndarray, faces: np.ndarray):
    """
    Creates a MeshLab MeshSet given a list of vertices and faces.
    """
    assert PYMESHLAB_AVAILABLE, "pymeshlab is not installed or could not be loaded."
    # Initialize MeshSet and create pymeshlab.Mesh
    mesh_set = pymeshlab.MeshSet()
    input_mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    mesh_set.add_mesh(input_mesh, "input_mesh")
    logging.info("Mesh successfully added to pymeshlab MeshSet.")
    return mesh_set


def cleanup(ms: pymeshlab.MeshSet):
    """
    General cleanup for a given Mesh. Removes degenerate elements from the
    geometry.
    """
    ms.meshing_remove_null_faces()
    ms.meshing_remove_folded_faces()
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_t_vertices()
    ms.meshing_remove_unreferenced_vertices()


def remove_floaters(ms: pymeshlab.MeshSet, threshold: float = 0.005):
    """
    Remove any floating artifacts that exist from our mesh generation.
    """
    assert PYMESHLAB_AVAILABLE, "pymeshlab is not installed or could not be loaded."
    ms.meshing_remove_connected_component_by_diameter(
        mincomponentdiag=pymeshlab.PercentageValue(15), removeunref=True
    )


def simplify_mesh(ms: pymeshlab.MeshSet, target_face_num: int):
    """
    Simplify the mesh to the target number of faces.
    """
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_face_num,
        qualitythr=0.4,
        preservenormal=True,
        autoclean=True,
    )


def save_mesh(ms: pymeshlab.MeshSet, output_path: str):
    """
    Save the mesh to a file.
    """
    ms.save_current_mesh(output_path)
    logging.info(f"Mesh saved to {output_path}.")


def postprocess_mesh(ms: pymeshlab.MeshSet, target_face_num: int, output_path: str):
    """
    Postprocess the mesh to the target number of faces.
    """
    cleanup(ms)
    remove_floaters(ms)
    simplify_mesh(ms, target_face_num)
