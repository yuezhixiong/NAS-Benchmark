import numpy as np
import torch


class MinNormSolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ( (v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2) )
        cost = v2v2 + gamma*(v1v2 - v2v2)
        return gamma, cost


    def _min_norm_2d(dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e8
        for i in range(dps.size()[0]): # task loop
            for j in range(i+1,dps.size()[0]):
                # if (i,j) not in dps:
                #     dps[(i, j)] = 0.0
                #     for k in range(len(vecs[i])):
                #         dps[(i,j)] += torch.dot(vecs[i][k], vecs[j][k]).data[0]
                #     dps[(j, i)] = dps[(i, j)]
                # if (i,i) not in dps:
                #     dps[(i, i)] = 0.0
                #     for k in range(len(vecs[i])):
                #         dps[(i,i)] += torch.dot(vecs[i][k], vecs[i][k]).data[0]
                # if (j,j) not in dps:
                #     dps[(j, j)] = 0.0   
                #     for k in range(len(vecs[i])):
                #         dps[(j, j)] += torch.dot(vecs[j][k], vecs[j][k]).data[0]
                c,d = MinNormSolver._min_norm_element_from2(dps[i,i], dps[i,j], dps[j,j])
                if d < dmin:
                    dmin = d
                    sol = [(i,j),c,d]
        return sol

    def _projection2simplex(y):
        """
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = len(y)
        # sorted_y = np.flip(np.sort(y), axis=0)
        sorted_y = torch.sort(y, descending=True)[0]
        tmpsum = 0.0
        tmax_f = (torch.sum(y) - 1.0)/m
        for i in range(m-1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1)/ (i+1.0)
            if tmax > sorted_y[i+1]:
                tmax_f = tmax
                break
        return torch.max(y - tmax_f, torch.zeros([y.size()[0], ]).cuda())
    
    def _next_point(cur_val, grad, n):
        proj_grad = grad - ( torch.sum(grad) / n )
        tm1 = -1.0*cur_val[proj_grad<0]/proj_grad[proj_grad<0]
        tm2 = (1.0 - cur_val[proj_grad>0])/(proj_grad[proj_grad>0])
        
        skippers = torch.sum(tm1<1e-7) + torch.sum(tm2<1e-7)
        t = torch.Tensor([1]).cuda()
        if (tm1[tm1>1e-7]).size()[0] > 0:
            t = torch.min(tm1[tm1>1e-7])
        if (tm2[tm2>1e-7]).size()[0] > 0:
            t = torch.min(t, torch.min(tm2[tm2>1e-7]))

        next_point = proj_grad*t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    def find_min_norm_element(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        """
        # Solution lying at the combination of two points


        vecs_clone = []
        for i in range(len(vecs)):
#             assert len(vecs[i]) == 1
            for k in range(len(vecs[i])):
                vecs_clone.append(vecs[i][k].view(-1).unsqueeze(0))
        vecs_clone = torch.cat(vecs_clone)

        grad_mat = torch.matmul(vecs_clone, vecs_clone.t())



        # dps = {}
        init_sol = MinNormSolver._min_norm_2d(grad_mat)
        
        n = len(vecs)
        sol_vec = torch.zeros([n,]).cuda()
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]
#         sol_vec = sol_vec.unsqueeze(0)

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec , init_sol[2]
    
        iter_count = 0

        # grad_mat = np.zeros((n,n))
        # for i in range(n):
        #     for j in range(n):
        #         grad_mat[i,j] = dps[(i, j)]
                

        while iter_count < MinNormSolver.MAX_ITER:
            grad_dir = -1.0 * torch.matmul(grad_mat, sol_vec)
#             sol_vec = sol_vec.squeeze()
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)
#             sol_vec = sol_vec.unsqueeze(0)
#             print(sol_vec.size(), new_point.size())
#             new_point = new_point.unsqueeze(0)
            # Re-compute the inner products for line search
            # v1v1 = 0.0
            # v1v2 = 0.0
            # v2v2 = 0.0
            # for i in range(n):
            #     for j in range(n):
            #         v1v1 += sol_vec[i]*sol_vec[j]*dps[(i,j)]
            #         v1v2 += sol_vec[i]*new_point[j]*dps[(i,j)]
            #         v2v2 += new_point[i]*new_point[j]*dps[(i,j)]
            
#             print('-'*10)
#             print(sol_vec.size(), new_point.size())
#             print(sol_vec.repeat(1, n).size())
#             print(new_point.repeat(n, 1).size())
            
            v1v1 = torch.sum(sol_vec.unsqueeze(1).repeat(1, n)*sol_vec.unsqueeze(0).repeat(n, 1)*grad_mat)
            v1v2 = torch.sum(sol_vec.unsqueeze(1).repeat(1, n)*new_point.unsqueeze(0).repeat(n, 1)*grad_mat)
            v2v2 = torch.sum(new_point.unsqueeze(1).repeat(1, n)*new_point.unsqueeze(0).repeat(n, 1)*grad_mat)

            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc*sol_vec + (1-nc)*new_point
            change = new_sol_vec - sol_vec
            if torch.sum(torch.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

    def find_min_norm_element_FW(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the Frank Wolfe until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n=len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec , init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                grad_mat[i,j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))

            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]

            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc*sol_vec
            new_sol_vec[t_iter] += 1 - nc

            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec


def gradient_normalizers(grads, losses, normalization_type):
    gn = {}
    if normalization_type == 'l2':
        for t in grads:
            gn[t] = np.sqrt(np.sum([gr.pow(2).sum().item() for gr in grads[t]]))
    elif normalization_type == 'loss':
        for t in grads:
            gn[t] = losses[t]
    elif normalization_type == 'loss+':
        for t in grads:
            gn[t] = losses[t] * np.sqrt(np.sum([gr.pow(2).sum().item() for gr in grads[t]]))
    elif normalization_type == 'none':
        for t in grads:
            gn[t] = 1.0
    else:
        print('ERROR: Invalid Normalization Type')
    return gn
