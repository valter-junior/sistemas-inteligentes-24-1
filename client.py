import connection as cn
import socket

s = cn.connect(3001)

estado, recompensa = cn.get_state_reward(s, "jump")
 