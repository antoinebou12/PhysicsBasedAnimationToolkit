import pbatoolkit as pbat
import pbatoolkit.fem
import pbatoolkit.geometry
import pbatoolkit.profiling
import pbatoolkit.math.linalg
import igl
import ipctk
import meshio
import numpy as np
import scipy as sp
import polyscope as ps
import polyscope.imgui as imgui
import math
import argparse
import itertools


def combine(V: list, C: list):
    Vsizes = [Vi.shape[0] for Vi in V]
    offsets = list(itertools.accumulate(Vsizes))
    C = [C[i] + offsets[i] - Vsizes[i] for i in range(len(C))]
    C = np.vstack(C)
    V = np.vstack(V)
    return V, C


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="3D elastic simulation of linear FEM tetrahedra using Incremental Potential Contact",
    )
    # parser.add_argument("-i", "--input", help="Paths to input mesh", nargs="+",
    #                     dest="inputs", required=True)
    parser.add_argument("-m", "--mass-density", help="Mass density", type=float,
                        dest="rho", default=1000.)
    parser.add_argument("-Y", "--young-modulus", help="Young's modulus", type=float,
                        dest="Y", default=1e6)
    parser.add_argument("-n", "--poisson-ratio", help="Poisson's ratio", type=float,
                        dest="nu", default=0.45)
    args = parser.parse_args()

    # Load input meshes and combine them into 1 mesh
    V, C = [], []
    # for input in args.inputs:
    #     imesh = meshio.read(args.input)
    #     V, C = imesh.points, imesh.cells_dict["tetra"]
    #     V = np.copy(V, order='C')
    #     C = C.astype(np.int64, order='C')

    # Test scenario, one fix bar on bottom, and another bar on top, rotated 90 degrees around the z axis
    imesh = meshio.read("data/bar.mesh")
    V.append(imesh.points.astype(np.float64, order='C'))
    C.append(imesh.cells_dict["tetra"].astype(np.int64, order='C'))
    R = sp.spatial.transform.Rotation.from_quat(
        [0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]).as_matrix()
    V2 = (V[0] - V[0].mean(axis=0)) @ R.T
    V2[:, 2] += (V[0][:, 2].max() - V[0][:, 2].min()) + 5*1e-3
    # V2[:, 2] += 1
    C2 = C[0]
    V.append(V2)
    C.append(C2)

    V, C = combine(V, C)
    mesh = pbat.fem.mesh(
        V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=1)
    V, C = mesh.X.T, mesh.E.T

    # Construct FEM quantities for simulation
    x = mesh.X.reshape(math.prod(mesh.X.shape), order='F')
    n = x.shape[0]
    v = np.zeros(n)
    X = V
    Xdot = v.reshape(mesh.X.shape[0], mesh.X.shape[1], order='F').T

    detJeM = pbat.fem.jacobian_determinants(mesh, quadrature_order=2)
    rho = args.rho
    M = pbat.fem.mass_matrix(mesh, detJeM, rho=rho,
                             dims=3, quadrature_order=2).to_matrix()
    # Lump mass matrix
    lumpedm = M.sum(axis=0)
    M = sp.sparse.spdiags(lumpedm, np.array([0]), m=M.shape[0], n=M.shape[0])
    Minv = sp.sparse.spdiags(
        1./lumpedm, np.array([0]), m=M.shape[0], n=M.shape[0])

    # Construct load vector from gravity field
    detJeU = pbat.fem.jacobian_determinants(mesh, quadrature_order=1)
    GNeU = pbat.fem.shape_function_gradients(mesh, quadrature_order=1)
    g = np.zeros(mesh.dims)
    g[-1] = -9.81
    fe = np.tile(rho*g[:, np.newaxis], mesh.E.shape[1])
    f = pbat.fem.load_vector(mesh, detJeU, fe, quadrature_order=1).to_vector()
    a = Minv @ f

    # Create hyper elastic potential
    Y = np.full(mesh.E.shape[1], args.Y)
    nu = np.full(mesh.E.shape[1], args.nu)
    psi = pbat.fem.HyperElasticEnergy.StableNeoHookean
    hep = pbat.fem.hyper_elastic_potential(
        mesh, detJeU, GNeU, Y, nu, psi=psi, quadrature_order=1)
    hep.precompute_hessian_sparsity()

    # Setup IPC contact handling
    F = igl.boundary_facets(C)
    F[:, :2] = np.roll(F[:, :2], shift=1, axis=1)
    E = ipctk.edges(F)
    cmesh = ipctk.CollisionMesh.build_from_full_mesh(V, E, F)
    dhat = 1e-3
    cconstraints = ipctk.CollisionConstraints()
    fconstraints = ipctk.FrictionConstraints()
    avgmass = lumpedm.mean()
    mu = 0.3
    epsv = 1e-4
    dmin = 1e-4
    BX = cmesh.map_displacements(X)
    BXdot = cmesh.map_displacements(Xdot)

    # Fix bottom of the input models as Dirichlet boundary conditions
    Xmin = mesh.X.min(axis=1)
    Xmax = mesh.X.max(axis=1)
    Xmax[-1] = Xmin[-1]+1e-4
    Xmin[-1] = Xmin[-1]-1e-4
    aabb = pbat.geometry.aabb(np.vstack((Xmin, Xmax)).T)
    vdbc = aabb.contained(mesh.X)
    dbcs = np.array(vdbc)[:, np.newaxis]
    dbcs = np.repeat(dbcs, mesh.dims, axis=1)
    for d in range(mesh.dims):
        dbcs[:, d] = mesh.dims*dbcs[:, d]+d
    dbcs = dbcs.reshape(math.prod(dbcs.shape))
    dofs = np.setdiff1d(list(range(n)), dbcs)

    ps.set_verbosity(0)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0.5)
    ps.set_program_name("Elasticity")
    ps.init()
    vm = ps.register_volume_mesh("world model", mesh.X.T, mesh.E.T)
    pc = ps.register_point_cloud("Dirichlet", mesh.X[:, vdbc].T)
    dt = 0.01
    animate = False
    newton_maxiter = 5
    dx = np.zeros(n)

    profiler = pbat.profiling.Profiler()
    ipctk.set_logger_level(ipctk.LoggerLevel.trace)

    def callback():
        global x, v, a, dx, hep, dt, M, f
        global cmesh, cconstraints, fconstraints, X, Xdot, BX, BXdot, dmin, dhat, mu, epsv, avgmass
        global newton_maxiter
        global animate, step
        global profiler

        changed, dt = imgui.InputFloat("dt", dt)
        changed, dhat = imgui.InputFloat(
            "IPC activation distance", dhat, format="%.6f")
        changed, mu = imgui.InputFloat(
            "Coulomb friction coeff", mu, format="%.2f")
        changed, newton_maxiter = imgui.InputInt(
            "Newton max iterations", newton_maxiter)
        changed, animate = imgui.Checkbox("animate", animate)
        step = imgui.Button("step")

        if animate or step:
            profiler.begin_frame("Physics")
            dt2 = dt**2
            xtilde = x + dt*v + dt2*a
            xk = x
            vk = v
            # Newton solve
            for k in range(newton_maxiter):
                # Compute collision constraints
                cconstraints.build(cmesh, BX, dhat, dmin=dmin)
                is_intersecting = ipctk.has_intersections(cmesh, BX)
                if is_intersecting:
                    print("Mesh is self intersecting!")
                    break
                gradB = cconstraints.compute_potential_gradient(
                    cmesh, BX, dhat)
                gradB = cmesh.to_full_dof(gradB)
                hessB = cconstraints.compute_potential_hessian(cmesh, BX, dhat)
                hessB = cmesh.to_full_dof(hessB)

                # Compute elasticity
                hep.compute_element_elasticity(xk, grad=True, hess=True)
                gradU, HU = hep.to_vector(), hep.to_matrix()

                # Compute adaptive barrier stiffness
                bboxdiag = ipctk.world_bbox_diagonal_length(BX)
                kB, maxkB = ipctk.initial_barrier_stiffness(
                    bboxdiag, dhat, avgmass, gradU, gradB, dmin=dmin)
                dprev = cconstraints.compute_minimum_distance(cmesh, BX)

                # Compute lagged friction constraints
                fconstraints.build(cmesh, BX, cconstraints, dhat, kB, mu)
                gradF = fconstraints.compute_potential_gradient(
                    cmesh, BXdot, epsv)
                gradF = cmesh.to_full_dof(gradF)
                hessF = fconstraints.compute_potential_hessian(
                    cmesh, BXdot, epsv)
                hessF = cmesh.to_full_dof(hessF)

                # Setup and solve for Newton search direction
                global bd, Add
                def setup():
                    global bd, Add
                    A = M + dt2 * HU + kB * hessB + hessF
                    b = -(M @ (xk - xtilde) + dt2*gradU + kB * gradB + gradF)
                    Add = A.tocsc()[:, dofs].tocsr()[dofs, :]
                    bd = b[dofs]

                profiler.profile("Setup Linear System", setup)

                if k > 0:
                    gradnorm = np.linalg.norm(bd, 1)
                    if gradnorm < 1e-4:
                        break

                def solve():
                    global dx, Add, bd
                    Addinv = pbat.math.linalg.ldlt(Add)
                    Addinv.compute(Add)
                    dx[dofs] = Addinv.solve(bd).squeeze()

                profiler.profile("Solve Linear System", solve)

                # CCD step truncation
                BXt0 = BX
                BXt1 = BX + cmesh.map_displacements(
                    dx.reshape(mesh.X.shape[0],
                               mesh.X.shape[1], order='F').T)
                max_alpha = ipctk.compute_collision_free_stepsize(
                    cmesh,
                    BXt0,
                    BXt1,
                    broad_phase_method=ipctk.BroadPhaseMethod.HASH_GRID,
                    min_distance=dmin
                )
                dx *= max_alpha

                # Update Newton iterate
                vk = (xk - x) / dt
                xk = xk + dx
                X = xk.reshape(mesh.X.shape[0], mesh.X.shape[1], order='F').T
                Xdot = vk.reshape(
                    mesh.X.shape[0], mesh.X.shape[1], order='F').T
                BX = cmesh.map_displacements(X)
                BXdot = cmesh.map_displacements(Xdot)

                # Update barrier stiffness
                dcurrent = cconstraints.compute_minimum_distance(cmesh, BX)
                kB = ipctk.update_barrier_stiffness(
                    dprev, dcurrent, maxkB, kB, bboxdiag, dmin=dmin)
                dprev = dcurrent

            v = (xk - x) / dt
            x = xk
            profiler.end_frame("Physics")

            # Update visuals
            vm.update_vertex_positions(X)

    ps.set_user_callback(callback)
    ps.show()
