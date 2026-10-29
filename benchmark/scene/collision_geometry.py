"""Read composed geometry for collision planning and contact calculations."""


def mesh_geometry(prim):
    """Triangulate polygons and weld duplicate exported vertices, without decimation."""
    import numpy as np
    import trimesh
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(prim)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    faces, start = [], 0
    for count in mesh.GetFaceVertexCountsAttr().Get():
        polygon = indices[start:start + count]
        faces.extend((polygon[0], polygon[i], polygon[i + 1]) for i in range(1, count - 1))
        start += count
    result = trimesh.Trimesh(np.asarray(mesh.GetPointsAttr().Get()), faces, process=False)
    result.merge_vertices()
    return result


def triangulate_planning_mesh(prim):
    """Convert unsupported faces in a clone using the authored collider semantics."""
    import mapbox_earcut
    import numpy as np
    from pxr import UsdGeom, UsdPhysics

    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    if UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "convexHull":
        # Scanned fruit contains folded/self-crossing surface polygons, but its
        # authored physical collider is a hull of the vertices. Honor that
        # existing collider in the planning clone instead of projecting the scan.
        from trimesh.convex import convex_hull
        hull = convex_hull(points)
        mesh.GetPointsAttr().Set(hull.vertices.astype(np.float32))
        mesh.GetFaceVertexCountsAttr().Set([3] * len(hull.faces))
        mesh.GetFaceVertexIndicesAttr().Set(hull.faces.reshape(-1).tolist())
        return
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    output_indices, counts, offset = [], [], 0
    for count in mesh.GetFaceVertexCountsAttr().Get():
        face = indices[offset:offset+count]
        offset += count
        if count < 3:
            raise ValueError(f"Degenerate planning face at {prim.GetPath()}")
        # cuRobo already handles triangles/quads, including nonplanar scan
        # faces. Project only unsupported n-gons; flattening a folded quad can
        # create an apparent self-intersection and discard part of its surface.
        if count <= 4:
            counts.append(count)
            output_indices.extend(face.tolist())
            continue
        polygon = points[face]
        normal = np.cross(polygon, np.roll(polygon, -1, axis=0)).sum(axis=0)
        if np.linalg.norm(normal) <= 1e-12:
            raise ValueError(f"Degenerate planning polygon at {prim.GetPath()}")
        projected = np.delete(polygon, np.abs(normal).argmax(), axis=1)
        local = mapbox_earcut.triangulate_float64(
            np.ascontiguousarray(projected), np.array([count], dtype=np.uint32)).reshape(-1, 3)
        if len(local) != count-2:
            raise ValueError(f"Incomplete planning triangulation at {prim.GetPath()}")
        counts.extend([3] * len(local))
        for triangle in face[local]:
            a, b, c = points[triangle]
            if np.dot(np.cross(b-a, c-a), normal) < 0:
                triangle = triangle[[0, 2, 1]]
            output_indices.extend(triangle.tolist())
    mesh.GetFaceVertexCountsAttr().Set(counts)
    mesh.GetFaceVertexIndicesAttr().Set(output_indices)


def _ray_distances(mesh, origin, direction):
    """Intersect visual triangles without requiring a separate ray-tracing backend."""
    import numpy as np

    triangles = mesh.triangles
    edge1, edge2 = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
    p = np.cross(direction, edge2)
    determinant = np.einsum("ij,ij->i", edge1, p)
    valid = abs(determinant)>1e-12
    inverse = np.divide(1., determinant, out=np.zeros_like(determinant), where=valid)
    offset = np.asarray(origin)-triangles[:, 0]
    u = np.einsum("ij,ij->i", offset, p)*inverse
    q = np.cross(offset, edge1)
    v = q@direction*inverse
    distance = np.einsum("ij,ij->i", edge2, q)*inverse
    hit = valid & (u>=0) & (v>=0) & (u+v<=1) & (distance>0)
    return sorted(set(np.round(distance[hit], 7).tolist()))
