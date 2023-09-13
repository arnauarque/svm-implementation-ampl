# Parameters declaration
param d; # Dimension of input space
param n; # Number of instances
param x {1..n, 1..d}; # Input data
param y {1..n}; # Labels
param nu;

# Variables declaration
var alpha {1..n};

# Objective function
maximize objective_function:
	(sum {i in 1..n} alpha[i]) -
	0.5 * sum {i in 1..n, j in 1..n} alpha[i]*alpha[j]*y[i]*y[j]*
		sum {k in 1..d} x[i,k]*x[j,k];

# Constraints
subject to constraint_1:
	sum {i in 1..n} alpha[i]*y[i] = 0;

subject to constraint_2 {i in 1..n}:
	alpha[i] >= 0;

subject to constraint_3 {i in 1..n}:
	alpha[i] <= nu;
