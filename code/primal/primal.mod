# Parameters declaration
param d; # Dimension of input space
param n; # Number of instances
param x {1..n, 1..d}; # Input data
param y {1..n}; # Labels
param nu;

# Variables declaration
var gamma;
var w {1..d};
var s {1..n};

# Objective function
minimize objective_function: 
	0.5 * sum {i in 1..d} w[i]^2 + nu * sum {i in 1..n} s[i];

# Constraints
subject to constraint_1 {i in 1..n}: 
	y[i]*(sum {j in 1..d} (w[j]*x[i,j]) + gamma) + s[i] >= 1;

subject to constraint_2 {i in 1..n}: 
	s[i] >= 0;
