#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @version: 0.1
# @license: Apache Licence
# @Filename:    fff.py
# @Author:      chaidisheng
# @contact: chaidisheng@stumail.ysu.edu.cn
# @site: https://github.com/chaidisheng
# @software: PyCharm
# @Time:        2/21/20 4:46 AM
# @torch: tensor.method or torch.method(tensor)

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import time
import math
import torch
import torch.nn as nn
import numpy as np
from utils.utils import *
from skimage.measure import compare_psnr


def check_adjoint(A, At, x, seed=0):
    r""" check At = A' """

    torch.manual_seed(seed)
    x = torch.randn(x.shape)
    Ax = A(x)

    y = torch.randn(Ax.size())
    Aty = At(y)

    Ax_real = Ax[:, :, :, 0:1].permute(0, 1, 3, 2)
    Ax_imag = torch.neg(Ax[:, :, :, 1:2]).permute(0, 1, 3, 2)

    y_real = y[:, :, :, 0:1]
    y_imag = y[:, :, :, 1:2]

    inner_real_1 = torch.matmul(Ax_real, y_real) - torch.matmul(Ax_imag, y_imag)
    inner_imag_1 = torch.matmul(Ax_real, y_imag) + torch.matmul(Ax_imag, y_real)
    inner_Product_1 = torch.cat((inner_real_1, inner_imag_1), dim=3)

    inner_Product_2 = torch.einsum('...ii->...', [torch.matmul(x.permute(0, 1, 3, 2), Aty)])  # trace: matrix
    inner_Product = inner_Product_1[:, :, :, 0:1] - inner_Product_2
    inner_Product = torch.cat((inner_Product, inner_Product_1[:, :, :, 1:2]), dim=3)

    error = torch.sqrt(torch.sum(inner_Product**2, dim=3, keepdim=True))
    error /= torch.max(torch.sqrt(torch.sum(inner_Product_1**2, dim=3, keepdim=True)), torch.abs(inner_Product_2))

    assert error < 1e-9, "At is not the adjoint of A, check the definitions of these operators!"
    return -1


def set_defaults(A, At, x0, gradf, options):
    r"""
    TODO: Docstring for setdefaults: Fill in the dictionary of options with the default values.
    :returns: options
    """
    # a set of all vaild option
    vaild_options = {"max_iters", "tol", "verbose", "record_objective",
                     "record_iterates", "adaptive", "accelerate", "restart",
                     "backtrack", "stepsize_shrink", "window", "eps_n", "L", "tau",
                     "function", "string_header", "stop_rule", "stop_now"}
    # check that every option provided by user is vaild
    try:
        for item in options:
            if item not in vaild_options:
                print("Options don't contain item {}!".format(item))
    except ValueError as e:
        raise e

    # options["max_iters"] = options.get("max_iters", 1000)

    # max_iters: the maximum number of iterations
    if "max_iters" not in options:
        options["max_iters"] = 1000

    # tol: the relative decrease in the residuals before the method stops
    if "tol" not in options:
        options["tol"] = 1e-3

    # verbose: if 'true' then print status information on every iteration
    if "verbose" not in options:
        options["verbose"] = False

    # record_objective: if 'true' then evaluate objective at every iteration
    if "record_objective" not in options:
        options["record_objective"] = False

    # record_iterates: if 'true' then record iterates in list(cell array)
    if "record_iterates" not in options:
        options["record_iterates"] = False

    # adaptive: if 'true' then use adaptive method
    if "adaptive" not in options:
        options["adaptive"] = True

    # accelerate: if 'true' then use FISTA-type adaptive method
    if "accelerate" not in options:
        options["accelerate"] = False

    # restart: if 'true' then restart the acceleration of FISTA
    # this only has an effect when options["accelerate"] = True
    if "restart" not in options:
        options["restart"] = True

    # backtrack: if 'true' then use backtracking line search
    if "backtrack" not in options:
        options["backtrack"] = True

    # stepsize_shrink: coefficient used to shrink stepsize when backtracking
    # kicks in
    if "stepsize_shrink" not in options:
        # the adaptive method can expand the stepsize, so we choose an
        # aggressive value here
        options["stepsize_shrink"] = 0.2
        if not options["adaptive"] or options["accelerate"]:
            # if the stepsize is monotonically decreasing, we don't want to make it smaller than we need
            options["stepsize_shrink"] = 0.5

    # create a mode string that describes which variant of the method is used
    options["mode"] = options.get("mode", "plain")
    if options["adaptive"]:
        options["mode"] = "adaptive"

    if options["accelerate"]:
        if options["restart"]:
            options["mode"] = "accelerated(FISTA)+restart"
        else:
            options["mode"] = "accelerated(FISTA)"

    # w: the window to look back when evaluating the max for the line search
    if "window" not in options:
        options["window"] = 10

    # eps_r: epsilon to prevent ratio residual from dividing by zero
    if "eps_r" not in options:
        options["eps_r"] = 1e-8

    # eps_n: epsilon to prevent normalized residual from dividing by zero
    if "eps_n" not in options:
        options["eps_n"] = 1e-8

    # Lipschitz continuous: lipschitz constant for smooth term. only needed if
    # tau has been set, in which case we need to approximate L so that tau can
    # be computed
    if ("L" not in options or (options["L"] <= 0)) and ("tau" not in options or
                                                        (options["tau"] <= 0)):
        x1, x2 = torch.randn_like(x0), torch.randn_like(x0)
        gradf1, gradf2 = At(gradf(A(x1))), At(gradf(A(x2)))
        options["L"] = torch.norm(gradf1 - gradf2, p='fro', dim=(2, 3), keepdim=True)
        options["L"] /= torch.norm(x2 - x1, p='fro', dim=(2, 3), keepdim=True)
        options["L"] = torch.max(options["L"], torch.tensor(1e-6).to(x0.device))
        options["tau"] = 2./options["L"]/10.

        assert options["tau"] > 0, "Invalid step size: " + str(options["tau"].numpy())

    # set tau if L was set by user
    if ("tau" not in options) or (options["tau"] <= 0):
        options["tau"] = 1./options["L"]
    else:
        options["L"] = 1./options["tau"]

    # function: an optional function that is compute and stored after every iteration
    if "function" not in options:
        # this functions get evaluated an each iterations, and results are stored
        options["function"] = lambda x: 0.  # default set

    # options["function"] = options.get("function", lambda x: 0.)

    # string_header: append this string to beginning of all output
    if "string_header" not in options:
        options["string_header"] = ""

    # the code below is for stopping rules
    # the filed 'stop_now' is function that returns 'true' if the iteration
    # should be terminated. the field 'stop_rule' is string that allows the
    # user to easily choose default values for 'stop_now'. the default stopping
    # rule terminates when the relative residual gets small
    if "stop_now" not in options:
        options["stop_now"] = "custom"

    if "stop_rule" not in options:
        options["stop_rule"] = "hybrid_residual"

    if options["stop_rule"] == "residual":
        options["stop_now"] = lambda x1, iters, residual, normalized_residual, max_residual, options: \
            residual < options["tol"]

    if options["stop_rule"] == "iterations":
        options["stop_now"] = lambda x1, iters, residual, normalized_residual, max_residual, options: \
            iters > options["max_iters"]

    # stop when normalized residual is small
    if options["stop_rule"] == "normalized_residual":
        options["stop_now"] = lambda x1, iters, residual, normalized_residual, max_residual, options: \
            normalized_residual < options["tol"]

    # divide by residual at iteration k by maximum residual over all iterations
    # terminate when this gets small
    if options["stop_rule"] == "ratio_residual":
        options["stop_now"] = lambda x1, iters, residual, normalized_residual, max_residual, options: \
            residual/(max_residual + options["eps_r"]) < options["tol"]

    # default behavior: stop if either normailzed or ratio residual is small
    if options["stop_rule"] == "hybrid_residual":
        options["stop_now"] = lambda x1, iters, residual, normalized_residual, max_residual, options: \
            residual/(max_residual + options["eps_r"]) < options["tol"] or normalized_residual < options["tol"]

    assert "stop_now" in options,  "Invalid  choice for stopping rule" + options["stop_rule"]
    return options


def fasta(A, At, f, gradf, g, proxg, x0, **options):
    r"""TODO: Docstring for fasta.

    FBS Method: a handy forward-backward solver
    optional form: minimzie f(Ax) + g(x)
    proximal opterators: proxg(z, t) = argmin mu*g(x) + .5||x - z||^2
    optional technique:
    (1) adaptive
    (2) accelerated
    (3) restart
    (4) backtracking line search
    This method solves the problem
        minimize f(Ax) + g(x)
    Where A is a matrix, f is differentiable, and both f and g are convex.
    The algorithm is an adaptive/accelerated forward-backward splitting.
    The user supplies function handles that evaluate 'f' and 'g'.  The user
    also supplies a function that evaluates the gradient of 'f' and the
    proximal operator of 'g', which is given by
                 proxg(z,t) = argmin t*g(x)+.5||x-z||^2.

   Inputs:
     A     : A matrix (or optionally a function handle to a method) that
              returns A*x
     At    : The adjoint (transpose) of 'A.' Optionally, a function handle
              may be passed.
     gradf : A function of z, computes the gradient of f at z
     proxg : A function of z and t, the proximal operator of g with
              stepsize t.
     x0    : The initial guess, usually a vector of zeros
     f     : A function of x, computes the value of f
     g     : A function of x, computes the value of g
     opts  : An optional struct with options.  The commonly used fields
              of 'opts' are:
                maxIters : (integer, default=1e4) The maximum number of iterations
                                allowed before termination.
                tol      : (double, default=1e-3) The stopping tolerance.
                                A smaller value of 'tol' results in more
                                iterations.
                verbose  : (boolean, default=false)  If true, print out
                                convergence information on each iteration.
                recordObjective:  (boolean, default=false) Compute and
                                record the objective of each iterate.
                recordIterates :  (boolean, default=false) Record every
                                iterate in a cell array.
             To use these options, set the corresponding field in 'opts'.
             For example:
                       >> opts.tol=1e-8;
                       >> opts.maxIters = 100;

   Outputs:
     sol  : The approximate solution
     outs : A struct with convergence information
     opts : A complete struct of options, containing all the values
            (including defaults) that were used by the solver.

    For more details, see the FASTA user guide, or the paper "A field guide
    to forward-backward splitting with a FASTA implementation."

    Copyright: Tom Goldstein, 2014. - matlab
    Copyright: chaidishheng, 2019. - pytorch

    """
    # Hyperparameters: check preconditions, fill missing optional entries on 'options'
    if not isinstance(A, (int, float)):
        assert not isinstance(At, (int, float)), "If A is a function handle, then At must be a handle as well."

    if isinstance(A, (int, float)):
        At = lambda x: torch.matmul(A.permute(0, 1, 3, 2), x)
        A = lambda x: torch.matmul(A, x)

    # if user didn't pass this arg, then create it
    if not options:
        options = dict()

    # verify that At=A'
    # check_adjoint(A, At, x0)

    # fill default values for options
    options = set_defaults(A, At, x0, gradf, options)

    if options["verbose"]:
        print("{}FASTA:\tmode = {}\tmax_iters = {},\ttol = {:e}\n".format
              (options["string_header"], options["mode"], options["max_iters"], options["tol"]))

    # record some frequently used information from options
    tau1 = options["tau"]  # initial stepsize
    max_iters = options["max_iters"]  # maximum iterations before automatic termination
    window = options["window"]  # lookback window for non-montone line search

    # allocate memory
    residual = []  # residuals
    normalized_residual = []  # normalized residuals
    taus = []  # stepsizes
    fvals = []  # the value of 'f', the smooth objective term
    objective = []  # the value of the objective function (f + g)
    func_values = []  # values of the optional 'function' argument in 'options'
    iterates = []  # record iterable solution
    total_backtracks = 0  # how many times was backtracking activated ?
    backtrack_count = 0  # backtracks on this iterations

    # initialize array values
    x1 = x0
    d1 = A(x1)
    f1 = f(d1)
    fvals.append(f1)
    gradf1 = At(gradf(d1))

    # local variables
    best_objective_iterate = None
    x_accel1, d_accel1, alpha1 = None, None, None
    solution, output = None, None

    # initialize additional storage required for FISTA
    if options["accelerate"]:
        x_accel1 = x0
        d_accel1 = d1
        alpha1 = 1.

    # to handle non-monotonicity
    # stores the maximum value of the residual that has been seen, used to
    # evaluated stopping conditions
    max_residual = torch.tensor(float('-inf')).to(x0.device)
    # stores the best objective value that has been seen. used to return best
    # iterate, rather than last iterate
    min_objective_values = torch.tensor(float('inf')).to(x0.device)

    # if user has chosen to record objective, then record initial value
    if options["record_objective"]:  # record function values 
        objective.append(f1 + g(x0))

    # begin recording solve time
    start_time = time.time()

    # begin loop
    for iters in range(max_iters):
        # 0 -> index i, and 1 -> index i+1
        x0 = x1  # x_{i} <-- x_{i+1}
        gradf0 = gradf1  # gradf0 is subgradient of f(x_{i})
        tau0 = tau1  # tau_{i} <-- tau_{i+1}

        # FBS step: obtain x_{i+1} from x_{i}
        x1hat = x0 - tau0*gradf0
        x1 = proxg(x1hat, tau0)

        # Non-monotone backtracking line search
        dx = x1 - x0
        d1 = A(x1)
        f1 = f(d1)
        if options["backtrack"]:
            max_objective = max(fvals[max(iters - window, 0):max(iters, 1)])  # list
            backtrack_count = 0
            gradf0_t = gradf0.permute(0, 1, 3, 2)  # transpose einsum
            affine = torch.matmul(gradf0_t, dx)
            affine = torch.einsum('...ii->...', [affine])
            sum_square = 1./(2*tau0)*torch.einsum('...ij->...', dx**2)
            # the backtracking loop: (note) 1e-12 is to quench rounding errors
            while (f1 - 1e-12 > max_objective + affine + sum_square) and (backtrack_count < 20):
                tau0 = tau0*options["stepsize_shrink"]
                x1hat = x0 - tau0*gradf0  # redo the FBS
                x1 = proxg(x1hat, tau0)
                d1 = A(x1)
                f1 = f(d1)
                dx = x1 - x0
                backtrack_count += 1
            total_backtracks += backtrack_count

        if options["verbose"] and backtrack_count > 10:
            print("{0}\tWARNING: excessive backtracking {1} steps, current stepsize is {2}\n".format
                  (options["string_header"], backtrack_count, tau0.cpu().item()))

        # record convergence information
        taus.append(tau0)  # stepsize
        # estimate of the gradient, should be zero at solution
        residual.append(torch.norm(dx, p='fro', dim=(2, 3), keepdim=True)/tau0)
        max_residual = torch.max(max_residual, residual[iters])
        normalizer = torch.max(torch.norm(gradf0, p='fro', dim=(2, 3), keepdim=True),
                               torch.norm(x1 - x1hat, p='fro', dim=(2, 3), keepdim=True)/tau0) + options["eps_n"]
        # normalizer += options["eps_n"]
        # normalized residual: size of discrepancy between the two derivative
        # terms, divided by size of the terms
        normalized_residual.append(residual[iters] / normalizer) # float(normalizer)
        fvals.append(f1)
        # record functions values
        func_values.append(options["function"](x0))

        if options["record_objective"]:
            objective.append(f1 + g(x1))
            new_objective_value = objective[iters + 1]  # problem
        else:
            # use the residual to evalute quality of iterate if we don not have
            # objective
            new_objective_value = residual[iters]

        # record optimal values
        if options["record_iterates"]:
            iterates.append(x1)

        # methods is non-monotone: make sure to record best solution
        if new_objective_value < min_objective_values:
            best_objective_iterate = x1
            min_objective_values = new_objective_value

        # print output information
        if options["verbose"]:
            psnr = compare_psnr(torch_to_np(x1), torch_to_np(options['ori_img']), data_range=1)
            print("%s%d: residual = %0.4f, backtrack = %d, tau = %f, PSNR=%.2fdB." %
                  (options["string_header"], iters, residual[iters], backtrack_count, tau0, psnr))  # format
            if options["record_objective"]:
                print(", objective = {}\n".format(objective[iters + 1]))
            else:
                print("\n")

        # text stopping criteria
        # if we stop, then record information in the output struct
        if options["stop_now"](x1, iters, residual[iters], normalized_residual[iters],
                               max_residual, options) or (iters + 1 >= max_iters):

            output = {}  # record output information
            end_time = time.time()
            output["solve_time"] = end_time - start_time
            print(("time consuming: %.2f" % (end_time - start_time)))
            output["residuals"] = residual
            output["stepsizes"] = taus
            output["normalized_residuals"] = normalized_residual
            output["objective"] = objective
            output["func_values"] = func_values
            output["backtracks"] = total_backtracks
            output["L"] = options["L"]
            output["initial_stepsize"] = options["tau"]
            output["iteration_count"] = iters
            if not options["record_objective"]:
                output["objective"] = "Not Recorded"
            if options["record_iterates"]:
                output["iterates"] = iterates

            # optimization solution
            solution = best_objective_iterate

            if options["verbose"]:
                print("{}\tDone: time = {:0.3f} secs, iterations = {}\n".format
                      (options["string_header"], output["solve_time"], output["iteration_count"]))

        # compute stepsize needed for next iteration using
        # BB(Barzilai-Borwein)/spectral method
        if options["adaptive"] and not options["accelerate"]:
            gradf1 = At(gradf(d1))
            # delta_g, note that delta_x was recorded above during backtracking
            dg = gradf1 + (x1hat - x0)/tau0
            dx_t = dx.permute(0, 1, 3, 2)
            schur_prod = torch.matmul(dx_t, dg)
            schur_prod = torch.einsum('...ii->...', [schur_prod])
            # first BB stepsize rule
            tau_s = torch.einsum('...ij->...', dx**2)/schur_prod
            # alternate BB stepsize rule
            tau_m = schur_prod/torch.einsum('...ij->...', dg**2)
            tau_m = torch.max(tau_m, torch.tensor(0.).to(x0.device))

            # use adaptive combination of tau_s and tau_m
            if 2*tau_m > tau_s:
                tau1 = tau_m
            else:
                # experiment with this parameter
                tau1 = tau_s - 0.5*tau_m
            # make sure step is non-negative
            if (tau1 <= 0) or math.isinf(tau1) or math.isnan(tau1):
                # let tau grow, backtracking will kick in if stepsize is too big
                tau1 = 1.5*tau0

        # use FISTA-type acceleration
        if options["accelerate"]:
            # store the old iterates
            x_accel0 = x_accel1
            d_accel0 = d_accel1
            alpha0 = alpha1
            x_accel1 = x1
            d_accel1 = d1
            # check to see if the acceleration needs to be restart
            delta_x = x0 - x1
            delta_accel = x_accel1 - x_accel0
            delta_x_t = delta_x.permute(0, 1, 3, 2)
            inner_prod = torch.matmul(delta_x_t, delta_accel)
            inner_prod = torch.einsum('...ii->...', [inner_prod])
            if options["restart"] and (inner_prod > 0):
                alpha0 = 1.
            # calculate acceleration parameter
            alpha1 = (1. + torch.tensor(np.sqrt(1 + 4.*alpha0**2)) if not torch.is_tensor(np.sqrt(1 + 4.*alpha0**2))
                      else np.sqrt(1 + 4.*alpha0**2))/2
            # over relax/predict
            x1 = x_accel1 + (alpha0 - 1.)/alpha1*(x_accel1 - x_accel0)
            d1 = d_accel1 + (alpha0 - 1.)/alpha1*(d_accel1 - d_accel0)
            # compute the gradient needed on the next iteration
            gradf1 = At(gradf(d1))
            fvals.append(f(d1))
            tau1 = tau0

        if (not options["adaptive"]) and (not options["accelerate"]):
            gradf1 = At(gradf(d1))
            tau1 = tau0

    return solution, output, options
