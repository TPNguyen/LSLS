function [Newgnd] = relabemin(gnd, postive)

Newgnd = gnd;
Newgnd(gnd == postive) = 1;
Newgnd(gnd ~= postive) = -1;

end