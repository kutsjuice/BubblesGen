using CALiPPSO  
using Statistics, Random
using WriteVTK
using YAML
using Gmsh

Random.seed!(120) # Just for reproducibility

#= UNcomment the following lines if you want to use Gurobi Solver, for running the tests.
    Or adjust them to the solver of your choice
=#
# using Gurobi
# const tol_overlap=1e-8 # tolerance for identifying an Overlap. Given that Gurobi's precision is 10^-9, a larger value is needed.
# const tol_optimality = 1e-9; # optimality tolerances. This is the most precise value allowed by Gurobi
# const grb_env = Gurobi.Env()
# const optimizer = Gurobi.Optimizer(grb_env)
# const solver_attributes = Dict("OutputFlag" => 0, "FeasibilityTol" => tol_optimality, "OptimalityTol" => tol_optimality, "Method" => 3, "Threads" =>  CALiPPSO.max_threads)


#= Comment the following lines if you want to use CALiPPSO with a different solver than GLPK; e.g., Gurobi (see above)=#
const optimizer = CALiPPSO.default_solver.Optimizer()
const tol_overlap = CALiPPSO.default_tol_overlap
const tol_optimality = CALiPPSO.default_tol_optimality
const solver_attributes = CALiPPSO.default_solver_attributes    


precompile_main_function(optimizer, solver_attributes)

const Nrand = 40 # size of configurations to test
const L = 1.0 # size of each side of the system's volume
const ds = 3 # dimensions to use
const ϕs_ds = 0.3


# ϕ = ϕs_ds

# r0, Xs0 = generate_random_configuration(ds, Nrand, ϕ, L); 

# points = Matrix{Float64}(undef, 3, Nrand)
# for i in 1:Nrand
#     points[1,i] = Xs0[i][1].value 
#     points[2,i] = Xs0[i][2].value 
#     points[3,i] = Xs0[i][3].value 
# end

# cells = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:Nrand]

# # vtk_grid("balls2", points, cells) do vtk
# #     vtk["radius"] = ones(Nrand)*r0
# # end


# # Testing CALiPPSO in monodisperse systems of different dimensions.
# # The initial conditions are random configurations of low density, i.e. very far from φ_J. 
# # The initial values of φ we're using have been chosen in such a way that the desired configuration can be created with ease
# for (nd, d) in enumerate(ds)
ϕ = ϕs_ds
d = ds

r0, Xs0 = generate_random_configuration(d, Nrand, ϕ, L); 
ℓ0 = 6*r0

printstyled("\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n", color=:blue)
printstyled("Using CALiPPSO to jam a system of N = ", Nrand, " in d = ", d, "\t Initial (φ, R) = ", [ϕ, r0], "\n", bold=:true, color=:blue)
printstyled("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n", color=:blue)

if d==5
    max_iters =5000
else
    max_iters = 1000
end

@time jammed_packing, info_convergence, Γs_vs_t, smax_vs_t, iso_vs_t = produce_jammed_configuration!(Xs0, r0; ℓ0=ℓ0, sqrΓ0=1.5,  max_iters=max_iters, initial_monitor=20, optimizer=optimizer, solver_attributes=solver_attributes) 
println("_______________________________________________________________________________________\n\n")
times = info_convergence.times_LP_optim
println("Info about LP times")
# println(times)
println("(min, avg ± std, max) LP times:\t", minimum(times), ", \t", mean(times), " ± ", std(times), ", \t", maximum(times))
println("_______________________________________________________________________________________\n\n\n")


res = network_of_contacts(jammed_packing)
points = Matrix{Float64}(undef, 3, length(jammed_packing.Particles))
for i in eachindex(jammed_packing.Particles)
    points[1,i] = jammed_packing.Particles[i].X[1].value
    points[2,i] = jammed_packing.Particles[i].X[2].value
    points[3,i] = jammed_packing.Particles[i].X[3].value
end
#
# cells = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in eachindex(jammed_packing.Particles)]

# vtk_grid("configuration", points, cells) do vtk

#     vtk["radius"] = ones(length(jammed_packing.Particles))*jammed_packing.R
# end


#

"""
Algorithm
1. Generate random monodisperce jamming configuration. This configuration gives the limit radius, before which sperec don't touch each other, and if spheres bigger - they are touching
2. Varying radius of spheres we can obtain different porosity of the cells. We will vary the radius from to both diraction.
3. Alternatevely, we can specify some distribution of radii, and varying mean value of radis we will obtain different porosity

"""




#

function prepare_to_periodic_sphere(xc, yc, zc, r, cs)
    centers = [[xc, yc, zc]];
    
    if abs(cs/2 - xc) < r
        push!(centers, [xc - cs, yc, zc])
    elseif abs(-cs/2 - xc) < r
        push!(centers, [xc + cs, yc, zc])
    end
    n = length(centers)
    if abs(cs/2 - yc) < r
        for i in 1:n
            pc = centers[i]
            push!(centers, [pc[1], pc[2] - cs, pc[3]])
        end
    elseif abs(-cs/2 - yc) < r
        for i in 1:n
            pc = centers[i]
            push!(centers, [pc[1], pc[2] + cs, pc[3]])
        end
    end
    n = length(centers)
    if abs(cs/2 - zc) < r
        for i in 1:n
            pc = centers[i]
            push!(centers, [pc[1], pc[2],pc[3] - cs])
        end
    elseif abs(-cs/2 - zc) < r
        for i in 1:n
            pc = centers[i]
            push!(centers, [pc[1], pc[2], pc[3] + cs])
        end
    end
    return centers
end

function create_plate_selection_bbox(dir, pos, size, eps = 1e-3)
    
    mask = [1,2,3] .== dir;
    bbox_start = Vector{Float64}(undef, 3);
    bbox_end = Vector{Float64}(undef, 3);

    bbox_start[mask] .= pos;
    bbox_start[.!mask] .= -size/2;
    bbox_start .-= eps;
    
    bbox_end[mask] .= pos;
    bbox_end[.!mask] .= size/2;
    bbox_end .+= eps;

    bbox = [
        bbox_start;
        bbox_end;
    ]
    return bbox;
end

function generate_cell_with_monodispers_pores(centers, radius; order = 1, subdivalg = 0)
    gmsh.initialize()
    gmsh.clear()
    box = gmsh.model.occ.addBox(-L/2, -L/2,-L/2,L,L,L)

    balls = []
   

    for point in eachcol(centers)
        pts_with_periodic = prepare_to_periodic_sphere(point..., radius, L)
        for pts in pts_with_periodic
            ball = gmsh.model.occ.addSphere(pts..., radius)
            push!(balls, (3,ball))
        end
    end

    gmsh.model.occ.synchronize()

    gmsh.model.occ.cut([(3, box)], balls)
    gmsh.model.occ.synchronize()

    ent = gmsh.model.occ.getEntities(3)
    gmsh.model.addPhysicalGroup(ent[1][1], [ent[1][2]], 300, "cell")

    for d1 in 1:3
        f_bbox = create_plate_selection_bbox(d1, -L/2, L*2);
        s_bbox = create_plate_selection_bbox(d1, L/2, L*2);

        for dim in [0, 1, 2]

            first_dimtags = gmsh.model.occ.getEntitiesInBoundingBox(f_bbox..., dim)
            second_dimtags = gmsh.model.occ.getEntitiesInBoundingBox(s_bbox..., dim)

            first_tags = [dimtag[2] for dimtag in first_dimtags];
            second_tags = [dimtag[2] for dimtag in second_dimtags];

            gmsh.model.addPhysicalGroup(dim, first_tags, -1, "n$(d1)_first")
            gmsh.model.addPhysicalGroup(dim, second_tags, -1, "n$(d1)_second")
    end
    

end

    v_body = gmsh.model.occ.getMass(ent[1]...)
    v_cell = L^3
    φ = 1 - v_body/v_cell
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20);
    gmsh.option.setNumber("Mesh.MeshSizeMax", L/30);
    gmsh.option.setNumber("Mesh.MeshSizeMin", L/50);
    gmsh.option.setNumber("Mesh.MaxNumThreads3D", 8);
    gmsh.option.setNumber("Mesh.ElementOrder", order);
    gmsh.option.setNumber("Mesh.Algorithm3D", 9);
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", subdivalg);

    # gmsh.model.occ.synchronize()
    # gmsh.model.addPhysicalGroup(ent[1][1], [ent[1][2]])


    gmsh.model.mesh.generate(3)
    # if !("-nopopup" in ARGS)
    #     gmsh.fltk.run()
    # end

    cell_data = Dict(
        "porosity"    => φ,
        "spheres_num" => Nrand,
        "radius_mean" => radius,
        "radius_std"  => 0,
        "radius_distribution" => "determined"
    )

    subdir = "geometry"
    files = readdir(subdir)
    filename = "cell_000"
    if filename*".msh" in files
        i=1;
        while filename[1:end-3]*lpad(string(i), 3, '0')*".msh" in files
            i+=1;
        end
        filename = filename[1:end-3]*lpad(string(i), 3, '0');
    end
    gmsh.write("$(subdir)/$(filename).msh")
    gmsh.write("$(subdir)/$(filename).vtk")
    YAML.write_file("$(subdir)/$(filename).yml", cell_data)

    #
    gmsh.finalize()             
end
##
porosity = [0.3, 0.4, 0.5, 0.6]
# porosity = [0.55]

# v_ball(r) = 4/3 * π * r^3
# v_overall = L^3
# φ = v_balls/v_overal
radius_from_porosity(p) = (p * L^3 / 4 * 3 / π / Nrand) ^(1/3)
radii = radius_from_porosity.(porosity)
# radius_factors = [0.8, 0.85, 0.9, 0.95, 1.0, 1.02]
for radius in radii[end:end]
    # radius = jammed_packing.R * k
    points = Matrix{Float64}(undef, 3, length(jammed_packing.Particles))
    for i in eachindex(jammed_packing.Particles)
        points[1,i] = jammed_packing.Particles[i].X[1].value
        points[2,i] = jammed_packing.Particles[i].X[2].value
        points[3,i] = jammed_packing.Particles[i].X[3].value
    end
    points .-=L/2

    generate_cell_with_monodispers_pores(points, radius)
end