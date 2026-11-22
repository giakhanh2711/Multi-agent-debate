import numpy as np
from itertools import combinations

class SupportEnumeration:
    def __init__(self, strategies):
        self.strategies = np.array(strategies)

    def solve_indifference(self, indv_matrix, idx_row, idx_col, size):
        if len(idx_col) != len(idx_row):
            return None
        
        sub_matrix = indv_matrix[np.ix_(idx_row, idx_col)].T
        Ax = np.block([
            [sub_matrix, -np.ones((len(idx_col), 1))],
            [np.ones((1, len(idx_row))), np.zeros((1, 1))]
        ])
        b = np.zeros(len(idx_col) + 1)
        b[-1] = 1

        try:
            probs = np.linalg.solve(Ax, b)
            probs = probs[:-1]
            if np.any(probs < -1e-8):
                return None

            mixed_strategy = np.zeros(size)
            mixed_strategy[list(idx_row)] = probs
            return mixed_strategy / np.sum(mixed_strategy)
        except np.linalg.LinAlgError:
            return None


    def is_valid_ne(self, player1_matrix, player2_matrix, mxs1, mxs2, supp1, supp2, tol=1e-7):
        expected_u1 = player1_matrix @ mxs2
        expected_u2 = mxs1 @ player2_matrix
        best_payoff1 = np.max(expected_u1)
        best_payoff2 = np.max(expected_u2)

        for s in supp1:
            if abs(expected_u1[s] - best_payoff1) > tol:
                return False
        for s in supp2:
            if abs(expected_u2[s] - best_payoff2) > tol:
                return False

        if np.any(expected_u1 > best_payoff1 + tol) or np.any (expected_u2 > best_payoff2 + tol):
            return False

        return True


    def support_enumeration(self, payoff_matrix, tol=1e-8):
        matrix1 = payoff_matrix[:, :, 0]
        matrix2 = payoff_matrix[:, :, 1]
        nrows, ncols = matrix1.shape

        nes = []
        # Consider all possible sizes of support of player 1
        max_size = min(nrows, ncols)
        for size1 in range(1, max_size + 1):
            # Consider all possible combinations for each size
            for supp1 in combinations(range(nrows), size1):
                    for supp2 in combinations(range(ncols), size1):
                        # Player 2 indifference, solve mixed strategy for player 1
                        p = self.solve_indifference(matrix2.T, supp1, supp2, nrows)
                        if p is None:
                            continue

                        # Player 1 indifference, solve mixed strategy for player 2
                        q = self.solve_indifference(matrix1, supp2, supp1, ncols)
                        if q is None:
                            continue

                        if self.is_valid_ne(matrix1, matrix2, p, q, supp1, supp2, tol):
                            is_duplicate = False
                            for x, y in nes:
                                if np.allclose(p, x, atol=tol) and np.allclose(q, y, atol=tol):
                                    is_duplicate = True
                                    break
                            if not is_duplicate:
                                nes.append((p, q))

        return nes


    def print_nes(self, nes):
        print(f"Strategies: {self.strategies}")
        print(f"There are {len(nes)} Nash equilibria")

        for p, q in nes:
            pure_p = np.count_nonzero(np.isclose(p, 1.0)) == 1
            pure_q = np.count_nonzero(np.isclose(q, 1.0)) == 1

            if pure_p and pure_q:
                i = np.argmax(p)
                j = np.argmax(q)
                print(
                    f"({self.strategies[i]}, "
                    f"{self.strategies[j]})"
                )
            else:
                print(f"Player 1: {p}")
                print(f"Player 2: {q}")