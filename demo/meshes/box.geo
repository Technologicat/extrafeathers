// Gmsh 2.16.0 geometry file

xmin = 0.0;
xmax = 1.0;
ymin = 0.0;
ymax = 1.0;
Point(1) = {xmin, ymin, 0, 5 * 0.01};  // x, y, z, [mesh element size]
Point(2) = {xmin, ymax, 0, 5 * 0.01};
Point(3) = {xmax, ymax, 0, 5 * 0.05};
Point(4) = {xmax, ymin, 0, 5 * 0.05};
Line(1) = {1, 2};  // |^
Line(2) = {2, 3};  // ->
Line(3) = {3, 4};  // |v
Line(4) = {4, 1};  // -<

Physical Line("left") = {1};
Physical Line("top") = {2};
Physical Line("right") = {3};
Physical Line("bottom") = {4};

// define the 2D domain
Line Loop(5) = {1, 2, 3, 4};   // outer boundary: clockwise to have normal pointing out of the domain
Plane Surface(6) = {5};
// Because there are four physical boundaries defined above, the ID of the physical surface will be 5.
Physical Surface("structure") = {6};
