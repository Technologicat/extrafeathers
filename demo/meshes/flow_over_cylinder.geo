// Gmsh 2.16.0 geometry file

// Configuration

// For convergence at high Reynolds numbers, the element size at outflow should be made very large.
//
// Long explanation:
//
// At high Re, a domain of otherwise reasonable length is not long enough for
// the disturbance introduced by the obstacle to be sufficiently smoothed out by
// diffusion so that a "smooth outflow" boundary condition could be meaningfully
// applied. That is, nontrivial structure remains in the flow when it hits the
// outflow boundary. This will often trigger a sudden convergence failure midway
// through a simulation.
//
// By using large elements near the outflow, we force the discretization to
// smooth out the remaining structure in the flow artificially (in the part
// where we are least interested in the solution), so that when the fluid
// parcels hit the outflow boundary, the outflow boundary conditions come as
// less of an "impedance mismatch" against the computed solution, making it
// much less likely for spurious numerical oscillations to arise there.
//
// Keep in mind that at the end of the channel, the "smooth outflow" boundary
// conditions cause the dynamic pressure to forcibly decrease to zero over one
// layer of elements (due to the zero Dirichlet BC on the pressure on the
// outflow boundary).
//
// Therefore, it is useful to make this last layer of elements as thick as
// reasonable. But to prevent spurious oscillations in the interesting parts
// of the domain, we should increase the local element size slowly and smoothly,
// as advised in the classic book by Gresho & Sani.

// Below, `elsize_walls` sets the element size at the walls near the obstacle,
// on its downstream side. These are useful to increase the resolution near the
// obstacle.

// Low Re setup (~1e2 ... ~1e3)
//
// Keep in mind that the problem is highly diffusive; don't make the smallest
// element size too small, or be prepared to face the wrath of parabolic partial
// differential equations (Δt ∝ h²).
elsize_inflow = 0.02;
elsize_obstacle = 0.01;
elsize_walls = 0.02;
elsize_outflow = 0.08;

// // High Re setup (~1e4)
// elsize_inflow = 0.02;
// elsize_obstacle = 0.01 / 16;  // for Re ~ 1e5, needs at least `/ 16`.
// elsize_walls = 0.02;
// elsize_outflow = 0.08 * 2;

// --------------------------------------------------------------------------------

// create the bounding box
Point(1) = {0.0, 0.0,  0, 1.0};  // x, y, z, [mesh element size]
Point(2) = {0.0, 0.41, 0, 1.0};
Point(10) = {0.5, 0.41, 0, 1.0};
Point(3) = {2.2, 0.41, 0, 1.0};
Point(4) = {2.2, 0.0,  0, 1.0};
Point(11) = {0.5, 0.0, 0, 1.0};
Line(1) = {1, 2};
Line(2) = {2, 10};
Line(12) = {10, 3};
Line(3) = {3, 4};
Line(13) = {4, 11};
Line(4) = {11, 1};

// Physical names.
//
// We define these in the same order as the boundary tags in the solver,
// so that the integer IDs auto-generated by Gmsh match the tags specified in the
// solver (which connects those tags to the definitions of the boundary conditions).
// The IDs automatically start from 1, drawing from a shared ID pool for every
// "Physical" entity in the mesh.
//
// The meshio conversion to FEniCS supports only one tag per facet (latest wins),
// because the tags are collected into a MeshFunction on facets. Named boundaries
// are not supported by the conversion; the purpose of the names is to make the
// .geo/.msh files themselves more self-documenting.
Physical Line("inflow")     = {1};
Physical Line("walls")      = {2, 12, 13, 4};
Physical Line("outflow")    = {3};

// create the obstacle (cylinder)
x0 = 0.2;
y0 = 0.2;
r = 0.05;
Point(5) = {x0, y0, 0, 1.0};      // .
Point(6) = {x0 - r, y0, 0, 1.0};  // <
Point(7) = {x0, y0 - r, 0, 1.0};  // v
Point(8) = {x0 + r, y0, 0, 1.0};  // >
Point(9) = {x0, y0 + r, 0, 1.0};  // ^
Circle(5) = {6, 5, 7};            // < . v
Circle(6) = {7, 5, 8};            // v . >
Circle(7) = {8, 5, 9};            // > . ^
Circle(8) = {9, 5, 6};            // ^ . <
Physical Line("cylinder") = {8, 5, 6, 7};

// define the 2D domain
Line Loop(9) = {2, 12, 3, 13, 4, 1};   // outer boundary: clockwise to have normal pointing out of the flow domain
Line Loop(10) = {7, 8, 5, 6};  // hole boundary:  counterclockwise to have normal pointing out of the flow domain
Plane Surface(11) = {9, 10};
// Because there are four physical boundaries defined above, the ID of the physical surface will be 5.
Physical Surface("fluid") = {11};

// Let's mesh the obstacle part, too. This gives us a mesh with two subdomains,
// and matching elements on the boundary.
Plane Surface(12) = {-10};  // clockwise to have normal pointing out of the solid domain
Physical Surface("structure") = {12};  // The ID of this physical surface will be 6.

// set local mesh size
// Characteristic Length is renamed to MeshSize in Gmsh 4.x.
// https://gmsh.info/doc/texinfo/gmsh.html#Specifying-mesh-element-sizes

Characteristic Length {2, 1} = elsize_inflow;
Characteristic Length {6, 7, 8, 9} = elsize_obstacle;
Characteristic Length {10, 11} = elsize_walls;
Characteristic Length {3, 4} = elsize_outflow;

// Mesh.MeshSizeFromPoints = 0;
// Mesh.MeshSizeFromCurvature = 0;
// Mesh.MeshSizeExtendFromBoundary = 0;
// Field[1] = MathEval;
// // This doesn't work, because Gmsh distributes the mesh size on edges first,
// // and then generates the interior based on that. Since the size along the walls
// // is constant, the size in the interior will also be constant. We need to
// // increase the size also along the walls as x increases.
// //Field[1].F = "0.01 + 4.0*(y / 0.41)*(1.0 - y / 0.41)";
// Field[1].F = "(x - 0.2)^2 + (y - 0.2)^2";
// Background Field = 1;
