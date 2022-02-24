// Gmsh 2.16.0 geometry file

elsize_walls = 0.04;
elsize_extrares = 0.04;
elsize_obstacle = 0.01 / 2;

// --------------------------------------------------------------------------------

// create the bounding box
xmin = 0.0;
xmax = 0.4;
ymin = 0.0;
ymax = 0.8;
Point(1) = {xmin, ymin, 0, 1.0};  // x, y, z, [mesh element size]
Point(10) = {xmin, (ymin + ymax) / 2.0, 0, 1.0};
Point(2) = {xmin, ymax, 0, 1.0};
Point(3) = {xmax, ymax, 0, 1.0};
Point(11) = {xmax, (ymin + ymax) / 2.0, 0, 1.0};
Point(4) = {xmax, ymin, 0, 1.0};
Line(1) = {1, 10};
Line(12) = {10, 2};
Line(2) = {2, 3};
Line(3) = {3, 11};
Line(13) = {11, 4};
Line(4) = {4, 1};

Physical Line("top") = {2};
Physical Line("walls") = {1, 12, 3, 13};
Physical Line("bottom") = {4};

// create the obstacle (cylinder)
x0 = 0.2;
y0 = 0.1;
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
Physical Line("obstacle") = {8, 5, 6, 7};

// define the 2D domain
Line Loop(9) = {2, 3, 13, 4, 1, 12};   // outer boundary: clockwise to have normal pointing out of the flow domain
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

Characteristic Length {1, 2, 3, 4} = elsize_walls;
Characteristic Length {10, 11} = elsize_extrares;
Characteristic Length {6, 7, 8, 9} = elsize_obstacle;
