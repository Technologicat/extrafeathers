// Gmsh 2.16.0 geometry file

r = 0.25;

// right half-circle
x0 = 0.0 + r;
y0 = 0.0;
Point(3) = {x0, y0, 0, 1.0};
Point(4) = {x0 + r * Cos(-Pi/4),  y0 + r * Sin(-Pi/4),  0,  1.0};
Point(5) = {x0 + r * Cos(0),      y0 + r * Sin(0),      0,  1.0};
Point(6) = {x0 + r * Cos(Pi/2),   y0 + r * Sin(Pi/2),   0,  1.0};
Point(7) = {x0 + r * Cos(Pi),     y0 + r * Sin(Pi),     0,  1.0};  // point where the halves touch

// Now we need a straight line to complete the right side.
//
// Given our half-circle, where should the tip of the heart go, at 45 degrees?
// The 45 degree straight line ends at the point where the half-circle is at 45 degrees:
x1 = x0 + r * Cos(-Pi/4);
y1 = y0 + r * Sin(-Pi/4);

// The other point on the 45 degree straight line is at the tip of the heart.
// The tip is horizontally in the middle, so it has the same value of x as the point where the halves touch.
x2 = x0 + r * Cos(Pi);
xtip = x2;
// We now have the equation of the straight line:
//   y = a x + b,  a = 1
// and two points (x1, y1), (x2, y2) on that line:
//   y1 = x1 + b
//   y2 = x2 + b
// where b and y2 are unknown. Solving the first equation for b,
// and inserting into the second equation yields y2 as
//   b = y1 - x1
//   y2 = x2 + y1 - x1
ytip = x2 + y1 - x1;
Point(1) = {xtip, ytip, 0, 1.0};  // tip

Line(1) = {1, 4};

// If we wanted to separate the inlet, we could do this:
// inlet_length = r / 2;
// Point(2) = {xtip + inlet_length, ytip + inlet_length, 0, 1.0};
// Line(1) = {1, 2};  // inlet part of straight line
// Line(2) = {2, 4};  // non-inlet part of straight line

Circle(3) = {4, 3, 5};
Circle(4) = {5, 3, 6};
Circle(5) = {6, 3, 7};

// left half-circle
x0 = 0.0 - r;
y0 = 0.0;
Point(8)  = {x0, y0, 0, 1.0};
// Point(9)  = {x0 + r * Cos(0),       y0 + r * Sin(0),      0,  1.0};  // same point as 7
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
Physical Line("inflow_R") = {1};
Physical Line("inflow_L") = {9};
Physical Line("walls")    = {3, 4, 5, 6, 7, 8};

// define the 2D domain
Line Loop(10) = {9, 8, 7, 6, 5, 4, 3, 1};  // outer boundary: clockwise to have normal pointing out of the flow domain
Plane Surface(11) = {10};

// Because there are three physical boundaries defined above, the ID of the physical surface will be 4.
Physical Surface("fluid") = {11};

// set local mesh size
// Characteristic Length is renamed to MeshSize in Gmsh 4.x.
// https://gmsh.info/doc/texinfo/gmsh.html#Specifying-mesh-element-sizes
Characteristic Length {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} = 0.01;
