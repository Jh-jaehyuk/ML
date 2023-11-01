computer = 'X'
user = 'O'
empty = '-'

infinity = 1000


class AI(object):
    def __init__(self, board):
        self.board = board[:]
        self.at_score = []
        self.is_finish()
        self.get_coords()
        self.cnt = 0

    def is_full(self, board):
        for y in range(3):
            for x in range(3):
                if board[y][x] == empty:
                    return False
        return True

    def is_win(self, player, board):
        # 가로와 세로  체크
        for y in range(3):
            row, column = 0, 0
            for x in range(3):
                if board[y][x] == player:
                    row += 1
                if board[x][y] == player:
                    column += 1
            if row == 3 or column == 3:
                return True

        # 대각선 체크
        x, row, column = 2, 0, 0
        for y in range(3):
            if board[y][y] == player:
                row += 1
            if board[y][x] == player:
                column += 1
            x -= 1
        if row == 3 or column == 3:
            return True

        return False

    def is_finish(self):
        return (self.is_full(self.board) or
                self.is_win(computer, self.board) or
                self.is_win(user, self.board))

    def get_coords(self):
        coords = []
        for y in range(3):
            for x in range(3):
                if self.board[y][x] == empty:
                    coords.append((x, y))
        return coords

    def evaluate(self):
        if self.is_win(computer, self.board):
            return 1
        elif self.is_win(user, self.board):
            return -1
        else:
            return 0

    def get_best_coord(self):
        score = -100
        best_coord = None
        for best in self.at_score:
            if best[0] > score:
                score = best[0]
                best_coord = best[1]
        self.at_score.clear()
        return best_coord

    def fill_board(self, coord, player):
        x, y = coord
        self.board[y][x] = player

    def minimax(self, depth, alpha, beta, player):
        self.cnt += 1
        if alpha >= beta:
            if player == computer:
                return infinity
            else:
                return -infinity
        if self.is_finish():
            return self.evaluate()

        coords = self.get_coords()
        max_score = -infinity
        min_score = infinity
        for coord in coords:
            if player == computer:
                self.fill_board(coord, computer)
                score = self.minimax(depth + 1, alpha, beta, user)
                max_score = max(score, max_score)
                alpha = max(max_score, alpha)

                if depth == 0:
                    self.at_score.append((max_score, coord))
            else:
                self.fill_board(coord, user)
                score = self.minimax(depth + 1, alpha, beta, computer)
                min_score = min(score, min_score)
                beta = min(min_score, beta)

            self.board[coord[1]][coord[0]] = empty
            if score == infinity or score == -infinity:
                break

        if player == computer:
            return max_score
        else:
            return min_score
