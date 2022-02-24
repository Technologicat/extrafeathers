// Gmsh 2.16.0 geometry file

elsize_walls = 0.02;
elsize_obstacle = 0.01;

// --------------------------------------------------------------------------------

// create the bounding box
Point(1) = {0.0, 0.0, 0, 1.0};  // x, y, z, [mesh element size]
Point(2) = {0.0, 0.4, 0, 1.0};
Point(3) = {0.4, 0.4, 0, 1.0};
Point(4) = {0.4, 0.0, 0, 1.0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Physical Line("walls")      = {1, 2, 3, 4};

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
Physical Line("obstacle") = {8, 5, 6, 7};

// define the 2D domain
Line Loop(9) = {2, 3, 4, 1};   // outer boundary: clockwise to have normal pointing out of the flow domain
Line Loop(10) = {7, 8, 5, 6};  // hole boundary:  counterclockwise to have normal pointing out of the flow domain
Plane Surface(11) = {9, 10};
// Because there are two physical boundaries defined above, the ID of the physical surface will be 3.
Physical Surface("fluid") = {11};

// Let's mesh the obstacle part, too. This gives us a mesh with two subdomains,
// and matching elements on the boundary.
Plane Surface(12) = {-10};  // clockwise to have normal pointing out of the solid domain
Physical Surface("structure") = {12};  // The ID of this physical surface will be 6.

// set local mesh size
// Characteristic Length is renamed to MeshSize in Gmsh 4.x.
// https://gmsh.info/doc/texinfo/gmsh.html#Specifying-mesh-element-sizes

Characteristic Length {1, 2, 3, 4} = elsize_walls;
Characteristic Length {6, 7, 8, 9} = elsize_obstacle;
