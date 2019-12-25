from cvxopt import matrix, spmatrix, spdiag, solvers


class CvxOpt:
    def __init__(self):
        self.NUM_RRH = 0
        self.NUM_USR = 0
        self.H = matrix()
        self.__clear__()

    def __clear__(self):
        self._G = []
        self._hq = []
        self.c = matrix(0., (self.NUM_RRH * self.NUM_USR + 1, 1))
        self.c[0, 0] = 1
        self.H = matrix()

    def feed(self, h=matrix(), cof=matrix(), p=matrix(), theta=matrix(), num_rrh=0, num_usr=0):
        self.NUM_RRH = num_rrh
        self.NUM_USR = num_usr
        self.__clear__()
        self.H = h

        for k_ in range(self.NUM_USR):
            w = [[0 for i in range(self.NUM_USR + 2)] for k in range(self.NUM_RRH * self.NUM_USR)]
            for l in range(self.NUM_RRH):
                for k in range(self.NUM_USR):
                    if k == k_:
                        w[l * self.NUM_USR + k][0] = - cof[k_] * h[l, k_]
                    w[l * self.NUM_USR + k][k + 1] = h[l, k_] * -1.

            w.insert(0, [0 for i in range(self.NUM_USR + 2)])
            self._G += [matrix(w)]
            self._hq += [spmatrix([theta[k_]], [self.NUM_USR + 2 - 1], [0], (self.NUM_USR + 2, 1))]

        for l_ in range(self.NUM_RRH):
            sp_value = []
            sp_index_i = []
            sp_index_j = []
            for k in range(self.NUM_USR):
                sp_value.append(-1.)
                sp_index_i.append(1 + l_ * self.NUM_USR + k)
                sp_index_j.append(1 + k)
            P = spmatrix(sp_value, sp_index_i, sp_index_j, size=(self.NUM_RRH * self.NUM_USR + 1, 1 + self.NUM_USR))
            self._G += [P.T]
            self._hq += [spmatrix([p[l_]], [0], [0], (1 + self.NUM_USR, 1))]

        self._hq += [matrix(0., (1, self.NUM_USR * self.NUM_RRH + 1)).T]
        d = matrix(-1., (1, self.NUM_USR * self.NUM_RRH + 1))
        D = spdiag(d)
        self._G += [D.T]

    def solve(self):
        solvers.options['show_progress'] = False
        # [print(i) for i in self._hq]
        sol = solvers.socp(self.c, Gq=self._G, hq=self._hq)
        return sol

    def showParams(self, sol):

        print('H:')
        print(self.H)
        print('C:')
        print(self.c)
        print('G:')
        for i in self._G:
            print(i)
        print('Hq:')
        for i in self._hq:
            print(i)
