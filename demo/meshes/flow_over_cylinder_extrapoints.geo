// Gmsh 2.16.0 geometry file

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

elsize_inflow = 0.02;
elsize_obstacle = 0.01 / 8;
elsize_walls = 0.02;  // the extra control points, for increased resolution near the obstacle
elsize_outflow = 0.08 * 2;

// physical names
//
// We define these in the same order as the boundary tags in the solver,
// so that the auto-generated integer IDs match the tags specified in the
// solver (which connects those tags to the definitions of the boundary conditions).
// The IDs automatically start from 1, drawing from a shared ID pool for every
// "Physical" thing in the mesh.
//
// The meshio conversion to FEniCS supports only one tag per facet (latest wins),
// because the tags are collected into a MeshFunction on facets. Named boundaries
// are not supported by the conversion; the purpose of the names is to make the
// .geo/.msh files themselves more self-documenting.
Physical Line("inflow")     = {1};
Physical Line("walls")      = {2, 12, 13, 4};
Physical Line("outflow")    = {3};

// create the obstacle (cylinder)
Point(5) = {0.2, 0.2, 0, 1.0};
Point(6) = {0.15, 0.2, 0, 1.0};
Point(7) = {0.2, 0.15, 0, 1.0};
Point(8) = {0.25, 0.2, 0, 1.0};
Point(9) = {0.2, 0.25, 0, 1.0};
Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 8};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 6};
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
