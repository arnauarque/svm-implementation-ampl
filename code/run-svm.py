import sys
from os import system, mkdir
from os.path import isfile, isdir
from argparse import ArgumentParser
from shutil import copyfile
import random
import numpy as np
import pandas as pd
from time import time

# ------------------------------------------------------------------------------
# -- Functional procedures
# ------------------------------------------------------------------------------

# Setting seeds for the random number generators
sd = 48219287
random.seed(sd)
np.random.seed(sd)

# Outputs an error message
def err(msg):
    print('run-svm.py: error: %s' % msg)
    sys.exit(1)

# Outputs a welcome message indicating the purpose of this script
def welcome_msg():
    div = '------------------------------------------------'
    print(div, 
          '-- OTDM Laboratory 2 script',
          '-- Authors: Arnau Arqué and Daniel Esquina',
          div, 
          '-- Script to run SVMs with different data', 
          '-- and parametrizations', 
          div, sep = '\n', end = '\n\n')

# Checks the requirements that are necessary to run the script
def check_requirements(args):
    # Data directory exists
    if not isdir('./data/'):
        mkdir('./data/')
    # Results directory exists
    if not isdir('./results/'):
        mkdir('./results/')
    # We have either 'cv' or 'nu' established
    if not args.nu and not args.cv:
        err('At least one of "--nu" and "--cv" arguments must be set.')
    # k-CV with k > 0
    if args.cv and args.cv[0] <= 0: 
        err('First value of "--cv" argument must be a non-zero positive number.')
    if args.cv and args.cv[2] <= args.cv[1]:
        err('Second and third "--cv" value must conform a valid range.')
    # Conditions to run the test with synthetic dataset
    if args.data == 'synthetic':
        if not isfile('./data/train.txt') or not isfile('./data/test.txt'):
            err('"train.txt" and "test.txt" files must exist in data directory!')


# ------------------------------------------------------------------------------
# -- Preprocessing methods
# ------------------------------------------------------------------------------

# Preprocessing of a synthetic dataset created with the genrator of the project
def preprocess_synthetic(fname, type):
    print('Preprocessing synthetic dataset...')
    file = open(fname, mode='r')
    auxfile = open('./data/aux%s.txt' % type, mode='w')
    
    lines = [ line.replace('*', '') for line in file.readlines() ]
    n = len(lines)
    d = len(lines[0].split()) - 1
    
    header = str(n) + (' %d' % d if type == 'train' else '')
    
    auxfile.write(header+'\n')
    auxfile.writelines(lines)
    
    auxfile.close()
    file.close()

# Method to numerize 'vars' variables using a dummy approach
def numerize(df, vars):
    for var in vars:
        newVars = pd.get_dummies(df[var], prefix = var)
        df = pd.concat([df, newVars], axis = 1)
        df = df.drop(columns = vars)
    return df

# Preprocessing and converting a CSV dataset to an AMPL formatted dataset
def csv_to_apml(fname, fileID, p, targetIdx, targets, header, delimiter, catVars = []):
    print('Converting "%s" dataset from CSV to AMPL format...' % fname)
    # Loading
    data = pd.read_csv(fname, header = header, delimiter = delimiter)
    
    # Preprocessing 
    for col in data.columns:
        if data[col].isnull().any():
            print('\tDopping missings in "%s" column ... ', end = '')
            data = data.drop(columns = col)
            print('Instances remaining: %d' % data.shape[0])
    
    # One-hot encoding for categorical variables
    data = numerize(data, catVars)
    
    # Shuffle data
    data = data.sample(frac=1).reset_index(drop=True)
    
    # Converting to AMPL format
    n, d = data.shape[0], data.shape[1]-1
    ntr = round(n*(100-p)/100)
    nte = n - ntr
    target = { x:('1' if i else '-1') for i, x in enumerate(targets) }
    labels = data[targetIdx]
    data = data.drop(columns = targetIdx)
    
    # Training set
    with open('./data/auxtrain.txt', mode = 'w') as file: 
        file.write('%d %d\n' % (ntr, d))
        for i in range(ntr):
            row = data.iloc[i].to_string(index = False).replace(' ', '').replace('\n', ' ')
            file.write('%s %s\n' % (row, target[labels[i]]))
            
    # Testing set 
    with open('./data/auxtest.txt', mode = 'w') as file: 
        file.write('%d\n' % nte)
        for i in range(ntr,n):
            row = data.iloc[i].to_string(index = False).replace(' ', '').replace('\n', ' ')
            file.write('%s %s\n' % (row, target[labels[i]]))

# ------------------------------------------------------------------------------
# -- Getters
# ------------------------------------------------------------------------------

# Given the results of our AMPL 'run' programs, returns the attribute 'x' as 
# a list of numbers or a single number (depending on the type of the original 
# attribute)
def get_attr(x, xs):
    if not isinstance(xs, list):
        with open(xs, mode = 'r') as file:
            xs = file.readlines()
    attr = None
    i = 0
    while i < len(xs) and not xs[i].startswith(x + ' ='):
        i += 1
    if i == len(xs):
        print('[WARNING] \'%s\' attribute not found in results.' % x)
    else:
        xs = xs[i].split('=')[-1].split()
        attr = [ float(x) if '.' in x else int(x) for x in xs ]
        if len(attr) == 1: 
            attr = attr[0]
    return attr

# Given a list of weights and 'b' of the equation of a hyperplane 
# PI:<w,x> + b = 0, returns the equation in string format
def hyperplane_str(ws, b):
    hyperplane = '%.5f*x1' % ws[0]
    for i in range(1,len(ws)):
        hyperplane += (' + ' if ws[i] > 0 else ' - ') + '%.5f*x%d' % (abs(ws[i]), i+1)
    hyperplane += (' + ' if b > 0 else ' - ') + '%.5f = 0' % abs(b)
    return hyperplane

# ------------------------------------------------------------------------------
# -- Cross validation
# ------------------------------------------------------------------------------

# Performs k-CV with NU=2^i, i = [start, start+1, ..., end] and runs the AMPL 
# model 'script'.
# Returns the best NU according to the testing accuracy obtained in each 
# iteration of the CV process
def cv(k, start, end, ampl, script):
    print('Running %d-fold CV with nu=2**i, i = [%d..%d]...' % (k, start, end))
    tr = './data/auxtrain.txt'
    te = './data/auxtest.txt'
    tr_cpy = './data/auxtrain_copy.txt'
    te_cpy = './data/auxtest_copy.txt'
    out = './results/output_aux.txt'
    # Saving original train/test dataset
    copyfile(tr, tr_cpy)
    copyfile(te, te_cpy)
    
    # Reading original train dataset
    with open(tr, mode = 'r') as file:
        lines = file.readlines()
    
    # Retrieving header and shuffling data
    n, d = [ int(x) for x in lines[0].strip().split() ]
    lines = lines[1:]
    random.shuffle(lines)
    
    # Number of instances per fold
    kn = int(n/k)
    print('\t#instances per fold = %d' % kn)
    
    # Variable to store the accuracies obtained with each nu
    accuracies = dict()
    
    # Running k-CV for each value of nu
    for i in range(start, end+1):
        # Setting value of nu and initializing variable for storing accuracy
        nu = 2**i
        acc = 0
        print('\tnu = %.4f\t' % nu, end = '', flush = True)
        # k-fold CV
        for j in range(k):
            print('.', end = '', flush = True)
            # Subdividing train dataset into train-fold, test-fold
            test = ['%d\n' % kn] + lines[j*kn : j*kn+kn]
            train = ['%d %d\n' % (n-kn, d)] + lines[0:j*kn] + lines[j*kn+kn:]
            with open(tr, mode = 'w') as file: 
                file.writelines(train)
            with open(te, mode = 'w') as file: 
                file.writelines(test)
            # Running SVM and getting testing accuracy
            system('export NU=%s && %s %s > %s' % (nu, ampl, script, out))
            acc += get_attr('te_acc', out)
        # Computing mean of accuracies
        acc /= k
        accuracies[nu] = acc
        print(' acc = %.2f' % acc)
    
    # Getting nu with highest accuracy
    best_nu = max(accuracies, key = accuracies.get)
    print('Best nu = %.4f' % best_nu)
    
    # Restoring train/test dataset
    copyfile(tr_cpy, tr)
    copyfile(te_cpy, te)
    system('rm %s %s %s' % (tr_cpy, te_cpy, out))
    return best_nu

# ------------------------------------------------------------------------------
# -- Main program
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    
    tic = time()
    
    parser = ArgumentParser()
    parser.add_argument('--data', required = True, 
                        choices = ['synthetic', 'sonar'],
                        help = 'Data that will be used to run the SVM')
    parser.add_argument('--type', required = True, 
                        choices = ['primal', 'dual'],
                        help = 'Type of objective function')
    parser.add_argument('--cv', required = False,
                        nargs = 3,
                        metavar = ('k', 'ini_exp_range', 'end_exp_range'),
                        type = int,
                        help = 'k-fold Cross Validation for \'nu\' estimation')
    parser.add_argument('--nu', required = False, 
                        type = float,
                        help = 'nu value')
    args = parser.parse_args()
    
    ampl = '../../ampl_macos64/ampl'
    script = './%s/%s.run' % (args.type, args.type)
    output = './results/output-%s-%s.txt' % (args.data, args.type)
    nu = args.nu
    
    # Checking  conditions
    check_requirements(args)
    
    # Welcome message
    welcome_msg()
    
    # 1. Generate data / preprocess data
    print('Loading data...')
    
    #   1.1. Synthetic
    if args.data == 'synthetic':
        preprocess_synthetic('./data/train.txt', 'train')
        preprocess_synthetic('./data/test.txt', 'test')
    
    #   1.2. Other datasets
    elif args.data == 'sonar':
        csv_to_apml(fname = './data/sonar.csv', fileID = 'sonar', 
                    p = 10, 
                    targetIdx = 60, targets = ['M', 'R'],
                    header = None, delimiter = ',')
    
    # 2. Run solver / Store results
    if args.cv:
        nu = cv(args.cv[0], args.cv[1], args.cv[2], ampl, script)
    print('Running SVM...')
    system('export NU=%s && %s %s > %s' % (nu, ampl, script, output))
    
    # 3. Retrieve and show accuracies / hyperplanes / ...
    print('Retrieving results...')
    # Reading output
    with open(output, mode = 'r') as file:
        results = file.readlines()
    
    # Getting attributes
    nu = get_attr('nu', results)
    ws = get_attr('ws', results)
    b = get_attr('gamma' if args.type == 'primal' else 'b', results)
    tr_acc = get_attr('tr_acc', results)
    te_acc = get_attr('te_acc', results)
    hyperplane = hyperplane_str(ws, b)
    
    # Showing results
    print('\n[ RESULTS ]')
    print('\tnu =', nu)
    print('\thyperplane:', hyperplane)
    print('\ttr_acc =', tr_acc)
    print('\tte_acc =', te_acc)
    
    print('\nExecution time: %.2f secs.' % (time() - tic))
    
    # 4. Clean workspace
    system('rm ./data/auxtrain.txt')
    system('rm ./data/auxtest.txt')
    