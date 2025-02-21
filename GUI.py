import tkinter as tk
from tkinter import messagebox
from threading import Thread
import AI

class TicTacToeGUI:
    def __init__(self, master):
        self.master = master
        self.ai = AI.EnhancedTicTacToeAI()
        self.init_gui()
        self.start_training()
        self.game_active = False

    def init_gui(self):
        self.master.title("增强型井字棋AI")
        self.canvas = tk.Canvas(self.master, width=300, height=300, bg='white')
        self.canvas.pack(pady=20)
        self.draw_board()

        control_frame = tk.Frame(self.master)
        control_frame.pack(pady=10)

        self.train_btn = tk.Button(control_frame, text="开始训练", command=self.start_training)
        self.train_btn.pack(side=tk.LEFT, padx=5)

        self.play_btn = tk.Button(control_frame, text="开始游戏", state=tk.DISABLED,
                                command=self.start_game)
        self.play_btn.pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar()
        self.status_label = tk.Label(self.master, textvariable=self.status_var, font=('Arial', 12))
        self.status_label.pack()

        self.canvas.bind("<Button-1>", self.handle_click)

    def start_game(self):
        self.game_active = True
        self.ai.reset()
        self.canvas.delete("all")
        self.draw_board()
        self.status_var.set("你的回合（X）")
        self.play_btn.config(state=tk.DISABLED)
        self.train_btn.config(state=tk.DISABLED)

    def draw_board(self):
        for i in range(1, 3):
            self.canvas.create_line(i * 100, 0, i * 100, 300, width=2)
            self.canvas.create_line(0, i * 100, 300, i * 100, width=2)

    def start_training(self):
        self.train_btn.config(state=tk.DISABLED)
        self.status_var.set("训练中...")

        def training_thread():
            # 修改回调函数接收三个参数
            def callback(episode, avg_reward, win_rate):  # ← 添加win_rate参数
                self.master.after(0, lambda: self.status_var.set(
                    # 在状态显示中添加胜率
                    f"训练进度：{episode / 100:.1f}千局 | 平均回报：{avg_reward:.2f}"
                ))

            self.ai.train(episodes=10000, callback=callback)
            self.master.after(0, lambda: [
                self.play_btn.config(state=tk.NORMAL),
                self.status_var.set("训练完成！点击开始游戏")
            ])

        Thread(target=training_thread).start()


    def handle_click(self, event):
        if not self.game_active:
            return
        x_count = self.ai.board.count('X')
        o_count = self.ai.board.count('O')
        if x_count > o_count:
            return
        x = event.x // 100
        y = event.y // 100
        pos = y * 3 + x
        if 0 <= pos < 9 and self.ai.board[pos] == ' ':
            self.make_move(pos, 'X')
            if not self.check_game_over():
                self.master.after(500, self.ai_move)

    def ai_move(self):
        state = self.ai.get_state()
        action = self.ai.choose_action(state)
        if action is not None and 0 <= action < 9:
            self.make_move(action, 'O')
            self.check_game_over()

    def make_move(self, pos, player):
        if self.ai.board[pos] == ' ':
            self.ai.board[pos] = player
            self.draw_symbol(pos, player)
            self.update_status()

    def draw_symbol(self, pos, player):
        x = (pos % 3) * 100 + 50
        y = (pos // 3) * 100 + 50
        color = 'blue' if player == 'X' else 'red'
        if player == 'X':
            self.canvas.create_line(x - 30, y - 30, x + 30, y + 30, width=3, fill=color)
            self.canvas.create_line(x + 30, y - 30, x - 30, y + 30, width=3, fill=color)
        else:
            self.canvas.create_oval(x - 30, y - 30, x + 30, y + 30, outline=color, width=3)

    def check_game_over(self):
        if self.ai.check_win('X'):
            self.show_result("玩家获胜！")
            self.game_active = False
            return True
        elif self.ai.check_win('O'):
            self.show_result("AI获胜！")
            self.game_active = False
            return True
        elif self.ai.is_draw():
            self.show_result("平局！")
            self.game_active = False
            return True
        return False

    def show_result(self, msg):
        messagebox.showinfo("游戏结束", msg)
        self.play_btn.config(state=tk.NORMAL)
        self.train_btn.config(state=tk.NORMAL)
        self.game_active = False

    def update_status(self):
        x_count = self.ai.board.count('X')
        o_count = self.ai.board.count('O')
        status = "玩家回合（X）" if x_count == o_count else "AI回合（O）"
        self.status_var.set(f"{status} | X:{x_count} O:{o_count}")


if __name__ == "__main__":
    root = tk.Tk()
    game = TicTacToeGUI(root)
    root.mainloop()
