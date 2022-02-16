// Gmsh 2.16.0 geometry file

// create the bounding box
Point(1) = {0.0, 0.0,  0, 1.0};  // x, y, z, [mesh element size]
Point(2) = {0.0, 0.61, 0, 1.0};
Point(3) = {2.2, 0.61, 0, 1.0};
Point(4) = {2.2, 0.0,  0, 1.0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
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
Physical Line("walls")      = {2, 4};
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

// create another the obstacle (another cylinder)
x0 = 0.2;
y0 = 0.4;
r = 0.05;
Point(10) = {x0, y0, 0, 1.0};      // .
Point(11) = {x0 - r, y0, 0, 1.0};  // <
Point(12) = {x0, y0 - r, 0, 1.0};  // v
Point(13) = {x0 + r, y0, 0, 1.0};  // >
Point(14) = {x0, y0 + r, 0, 1.0};  // ^
Circle(9) = {11, 10, 12};          // < . v
Circle(10) = {12, 10, 13};         // v . >
Circle(11) = {13, 10, 14};         // > . ^
Circle(12) = {14, 10, 11};         // ^ . <

Physical Line("cylinder") = {8, 5, 6, 7,  12, 9, 10, 11};

// define the 2D domain
Line Loop(13) = {2, 3, 4, 1};   // outer boundary: clockwise to have normal pointing out of the flow domain
Line Loop(14) = {7, 8, 5, 6};  // hole boundary:  counterclockwise to have normal pointing out of the flow domain
Line Loop(15) = {11, 12, 9, 10};  // hole boundary
Plane Surface(16) = {13, 14, 15};
// Because there are four physical boundaries defined above, the ID of the physical surface will be 5.
Physical Surface("fluid") = {16};

// Fluid mesh only for now, because we don't want disconnected structure meshes.

// set local mesh size
// Characteristic Length is renamed to MeshSize in Gmsh 4.x.
// https://gmsh.info/doc/texinfo/gmsh.html#Specifying-mesh-element-sizes
Characteristic Length {6, 7, 8, 9, 11, 12, 13, 14} = 0.01;  // obstacle (cylinder) surfaces
Characteristic Length {2, 1} = 0.02;        // inflow corners
Characteristic Length {3, 4} = 0.1;         // outflow corners

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
