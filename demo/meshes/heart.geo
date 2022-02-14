// Gmsh 2.16.0 geometry file

r = 0.25;

// right half-circle
x0 = 0.0 + r;
y0 = 0.0;
Point(3) = {x0, y0, 0, 1.0};
Point(4) = {x0 + r * Cos(-Pi/4),  y0 + r * Sin(-Pi/4),  0,  1.0};
Point(5) = {x0 + r * Cos(0),      y0 + r * Sin(0),      0,  1.0};
Point(6) = {x0 + r * Cos(Pi/2),   y0 + r * Sin(Pi/2),   0,  1.0};
Point(7) = {x0 + r * Cos(Pi),     y0 + r * Sin(Pi),     0,  1.0};
x1 = x0 + r * Cos(-Pi/4);
y1 = y0 + r * Sin(-Pi/4);
x2 = x0 + r * Cos(Pi);

// Where do we enter the circle tangentially at 45 degrees?
// y = a x + b,  a = 1
//   y1 = x1 + b
//   y2 = x2 + b
// solve for b and then y2
//   b = y1 - x1
//   y2 = x2 + y1 - x1
p1x = x2;
p1y = x2 + y1 - x1;
// inlet_length = r / 2;
Point(1) = {p1x, p1y, 0, 1.0};  // tip
// Point(2) = {p1x + inlet_length, p1y + inlet_length, 0, 1.0};  // for separating the inlet

Line(1) = {1, 4};
// Line(2) = {2, 4};
Circle(3) = {4, 3, 5};
Circle(4) = {5, 3, 6};
Circle(5) = {6, 3, 7};

// left half-circle
x0 = 0.0 - r;
y0 = 0.0;
Point(8)  = {x0, y0, 0, 1.0};
// Point(9)  = {x0 + r * Cos(0),       y0 + r * Sin(0),      0,  1.0};
Point(10) = {x0 + r * Cos(Pi/2),    y0 + r * Sin(Pi/2),   0,  1.0};
Point(11) = {x0 + r * Cos(Pi),      y0 + r * Sin(Pi),     0,  1.0};
Point(12) = {x0 + r * Cos(5*Pi/4),  y0 + r * Sin(5*Pi/4),  0,  1.0};

Circle(6) = {7, 8, 10};
Circle(7) = {10, 8, 11};
Circle(8) = {11, 8, 12};
Line(9) = {12, 1};

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
Physical Line("inflow_R")     = {1};
Physical Line("inflow_L")     = {9};
Physical Line("walls")      = {3, 4, 5, 6, 7, 8};

// define the 2D domain
Line Loop(10) = {9, 8, 7, 6, 5, 4, 3, 1};  // outer boundary: clockwise to have normal pointing out of the flow domain
Plane Surface(11) = {10};

// Because there are two physical boundaries defined above, the ID of the physical surface will be 3.
Physical Surface("fluid") = {11};

// set local mesh size
// Characteristic Length is renamed to MeshSize in Gmsh 4.x.
// https://gmsh.info/doc/texinfo/gmsh.html#Specifying-mesh-element-sizes
Characteristic Length {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} = 0.01;
