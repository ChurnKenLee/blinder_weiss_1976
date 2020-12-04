function eq = lab_solve(l,params)
a = params(1);
z = params(2);
w = params(3);
r = params(4);
s = params(5);
Frisch = params(6);

eq = l - (w*z*l + r*a)^(-s*Frisch)*(w*z)^Frisch;