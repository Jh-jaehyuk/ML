import pygame, sys, random
from pygame.locals import *
from buildMinimax import *

width = 300
height = 400

center_x = width // 2
center_y = (height - 30) // 2
box_size = 80
text_size = 50

top = center_y - box_size - box_size // 2
down = top + box_size * 3
left = center_x - box_size - box_size // 2
right = left + box_size * 3

fps = 30
fps_clock = pygame.time.Clock()

white = (255, 255, 255)
black = (0, 0, 0)
blue = (0, 0, 255)


def main():
    pygame.init()
    surface = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Tic Tac Toe')
    surface.fill(white)
    menu = Menu(surface)
    ttt = TTT(surface, menu)
    while True:
        run_game(surface, menu, ttt)
        menu.is_continue()


def run_game(surface, menu, ttt):
    reset_game(surface, menu, ttt)
    while True:
        is_user = True
        if ttt.turn == computer:
            is_user = False
            ttt.play_computer()

        for event in pygame.event.get():
            if event.type == QUIT:
                terminate()
            elif event.type == MOUSEBUTTONUP:
                if is_user:
                    if not ttt.check_board(event.pos):
                        if menu.check_rect(event.pos):
                            reset_game(surface, menu, ttt)

        if ttt.check_gameover():
            return
        pygame.display.update()
        fps_clock.tick(fps)


def terminate():
    pygame.quit()
    sys.exit()


def reset_game(surface, menu, ttt):
    surface.fill(white)
    menu.draw_menu()
    ttt.init_game()


class TTT(object):
    def __init__(self, surface, menu):
        self.board = [['-' for i in range(3)] for j in range(3)]
        self.coords = []
        self.set_coords()
        self.surface = surface
        self.menu = menu
        pass

    def init_game(self):
        self.ai = AI(self.board)
        self.draw_line(self.surface)
        self.finish = False
        self.init_board()
        self.set_first_player()

    def set_first_player(self):
        self.turn = random.choice([computer, user])
        if self.turn == computer:
            x = random.choice([0, 1, 2])
            y = random.choice([0, 1, 2])
            self.draw_shape(x, y)

    def init_board(self):
        for y in range(3):
            for x in range(3):
                self.board[y][x] = empty

    def set_coords(self):
        for i in range(3):
            for j in range(3):
                coord = left + j * box_size, top + i * box_size
                self.coords.append(coord)

    def get_coord(self, pos):
        for coord in self.coords:
            x, y = coord
            rect = pygame.Rect(x, y, box_size, box_size)
            if (rect.collidepoint(pos)):
                return coord
        return None

    def get_board(self, coord):
        x, y = coord
        x = (x - left) // box_size
        y = (y - top) // box_size
        return x, y

    def play_computer(self):
        self.ai.minimax(0, -infinity, infinity, computer)
        print(self.ai.cnt)
        self.ai.cnt = 0
        x, y = self.ai.get_best_coord()
        self.draw_shape(x, y)

    def check_board(self, pos):
        coord = self.get_coord(pos)
        if not coord:
            return False

        x, y = self.get_board(coord)
        if self.board[y][x] != empty:
            return True
        else:
            return self.draw_shape(x, y)

    def draw_shape(self, x, y):
        self.board[y][x] = self.turn

        if self.ai.is_win(self.turn, self.board):
            self.finish = True
            self.menu.show_msg(self.turn)

        x, y = self.get_pixel_coord(x, y)
        if self.turn == computer:
            self.draw_x(x, y)
            self.turn = user
        else:
            self.draw_o(x, y)
            self.turn = computer
        return True

    def draw_line(self, surface):
        for i in range(1, 3, 1):
            gap = box_size * i
            pygame.draw.line(surface, black, (left, top + gap), (right, top + gap), 5)
            pygame.draw.line(surface, black, (left + gap, top), (left + gap, down), 5)

    def get_pixel_coord(self, x, y):
        x = left + x * box_size
        y = top + y * box_size
        return x, y

    def draw_o(self, x, y):
        half = box_size // 2
        r = text_size // 2
        pygame.draw.circle(self.surface, blue, (x + half, y + half), r, 5)

    def draw_x(self, x, y):
        x1, y1 = x + 15, y + 15
        x2, y2 = x1 + text_size, y1 + text_size
        pygame.draw.line(self.surface, blue, (x1, y1), (x2, y2), 7)
        pygame.draw.line(self.surface, blue, (x2, y1), (x1, y2), 7)

    def check_gameover(self):
        if self.ai.is_full(self.board):
            if not self.finish:
                self.menu.show_msg('tie')
            self.finish = True
        return self.finish


class Menu(object):
    def __init__(self, surface):
        self.font = pygame.font.Font('freesansbold.ttf', 20)
        self.surface = surface
        self.draw_menu()

    def draw_menu(self):
        x = center_x - width // 4
        self.new_rect = self.make_text('New Game', black, white, x, height - 30)
        x = center_x + width // 4
        self.quit_rect = self.make_text('Quit Game', black, white, x, height - 30)

    def show_msg(self, msg_id):
        msg = {
            'X': 'You lost!',
            'O': 'You win!!',
            'tie': 'Tie',
        }
        self.make_text(msg[msg_id], blue, white, center_x, 30)

    def make_text(self, text, color, bgcolor, cx, cy):
        surf = self.font.render(text, True, color, bgcolor)
        rect = surf.get_rect()
        rect.center = (cx, cy)
        self.surface.blit(surf, rect)
        return rect

    def check_rect(self, pos):
        if self.new_rect.collidepoint(pos):
            return True
        elif self.quit_rect.collidepoint(pos):
            terminate()
        return False

    def is_continue(self):
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    terminate()
                elif event.type == MOUSEBUTTONUP:
                    if (self.check_rect(event.pos)):
                        return
            pygame.display.update()
            fps_clock.tick(fps)


if __name__ == '__main__':
    main()