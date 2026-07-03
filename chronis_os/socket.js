const jwt = require('jsonwebtoken');
const db  = require('./db');

module.exports = function initSocket(io) {
  io.use((socket, next) => {
    const token = socket.handshake.auth?.token;
    if (!token) return next(new Error('Authentication required'));
    try {
      socket.user = jwt.verify(token, process.env.OS_JWT_SECRET);
      next();
    } catch {
      next(new Error('Invalid or expired token'));
    }
  });

  io.on('connection', async (socket) => {
    const { id: userId, role, name } = socket.user;

    socket.join(`user:${userId}`);
    socket.join(`role:${role}`);

    console.log(`[OS socket] ${name} (${role}) connected`);

    try {
      await db.query('UPDATE users SET last_active = NOW() WHERE id = $1', [userId]);
    } catch (e) { console.error('[OS socket] last_active update failed:', e.message); }

    socket.to('role:manager').to('role:cto').to('role:ceo').emit('user:presence', {
      userId, name, role, online: true,
    });

    const heartbeat = setInterval(async () => {
      try {
        await db.query('UPDATE users SET last_active = NOW() WHERE id = $1', [userId]);
      } catch (e) { console.error('[OS socket] heartbeat error:', e.message); }
    }, 30_000);

    socket.on('notifications:ping', async () => {
      try {
        const { rows } = await db.query(
          'SELECT COUNT(*) FROM notifications WHERE receiver_id = $1 AND read = FALSE',
          [userId]
        );
        socket.emit('notifications:count', { count: parseInt(rows[0].count, 10) });
      } catch (e) { console.error('[OS socket] ping error:', e.message); }
    });

    socket.on('disconnect', (reason) => {
      clearInterval(heartbeat);
      console.log(`[OS socket] ${name} disconnected (${reason})`);
      setTimeout(() => {
        const rooms = io.sockets.adapter.rooms.get(`user:${userId}`);
        if (!rooms || rooms.size === 0) {
          socket.to('role:manager').to('role:cto').to('role:ceo').emit('user:presence', {
            userId, name, role, online: false,
          });
        }
      }, 5000);
    });
  });

  io.sendNotification = (receiverId, notification) => {
    io.to(`user:${receiverId}`).emit('notification', notification);
  };

  io.kickUser = (userId, status = 'terminated') => {
    io.to(`user:${userId}`).emit('account_terminated', { status });
    const room = io.sockets.adapter.rooms.get(`user:${userId}`);
    if (room) {
      room.forEach(socketId => {
        const s = io.sockets.sockets.get(socketId);
        if (s) s.disconnect(true);
      });
    }
  };
};
