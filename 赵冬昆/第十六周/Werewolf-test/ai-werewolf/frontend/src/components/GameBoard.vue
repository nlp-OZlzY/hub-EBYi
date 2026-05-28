<template>
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
    <!-- 左侧：玩家区域 -->
    <div class="lg:col-span-2 space-y-6">
      <!-- 游戏状态 -->
      <div class="card p-4">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-4">
            <span class="text-gray-400">游戏 ID:</span>
            <span class="font-mono text-indigo-400">{{ game.gameId }}</span>
          </div>
          <div class="flex items-center gap-4">
            <span class="text-gray-400">第 {{ game.day }} 天</span>
            <span 
              :class="getPhaseClass(game.phase)" 
              class="phase-tag"
            >
              {{ game.phaseText }}
            </span>
          </div>
        </div>
      </div>
      
      <!-- 玩家卡片 -->
      <div class="card p-6">
        <h3 class="text-lg font-semibold mb-4 flex items-center gap-2">
          <svg class="w-5 h-5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"></path>
          </svg>
          玩家列表
        </h3>
        
        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          <div 
            v-for="player in game.players" 
            :key="player.id"
            :class="[
              'relative rounded-xl p-4 transition-all duration-300',
              player.isAlive ? 'bg-gray-800 hover:bg-gray-700' : 'bg-gray-900 opacity-50'
            ]"
          >
            <!-- 存活指示器 -->
            <div 
              :class="[
                'absolute -top-2 -right-2 w-5 h-5 rounded-full border-2 border-gray-800',
                player.isAlive ? 'bg-green-500 shadow-lg shadow-green-500/50' : 'bg-red-500'
              ]"
            ></div>
            
            <!-- 角色图标 -->
            <div class="text-4xl text-center mb-2">{{ player.avatar }}</div>
            
            <!-- 玩家信息 -->
            <div class="text-center">
              <div class="font-medium text-sm">{{ player.name }}</div>
              <div 
                v-if="game.isOver"
                :class="[
                  'text-xs mt-1',
                  getRoleColor(player.role)
                ]"
              >
                {{ getRoleText(player.role) }}
              </div>
              <div v-else class="text-xs text-gray-500 mt-1">???</div>
            </div>
            
            <!-- 死亡遮罩 -->
            <div 
              v-if="!player.isAlive"
              class="absolute inset-0 bg-black/50 rounded-xl flex items-center justify-center"
            >
              <svg class="w-6 h-6 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
              </svg>
            </div>
          </div>
        </div>
      </div>
      
      <!-- 对话区域 -->
      <div class="card p-6">
        <h3 class="text-lg font-semibold mb-4 flex items-center gap-2">
          <svg class="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"></path>
          </svg>
          对话记录
        </h3>
        
        <div class="space-y-3 max-h-64 overflow-y-auto">
          <div 
            v-for="(dialogue, index) in game.dialogues.slice(-10)" 
            :key="index"
            class="bg-gray-800/50 rounded-lg p-3"
          >
            <div class="flex items-center justify-between mb-1">
              <span class="font-medium text-sm">{{ dialogue.speaker }}</span>
              <span class="text-xs text-gray-500">{{ dialogue.time }}</span>
            </div>
            <p class="text-sm text-gray-300">{{ dialogue.content }}</p>
          </div>
          
          <div v-if="game.dialogues.length === 0" class="text-center text-gray-500 py-8">
            暂无对话记录
          </div>
        </div>
      </div>
      
      <!-- 操作按钮 -->
      <div class="card p-4">
        <div class="flex items-center justify-between">
          <div v-if="game.stepData && game.stepData.eliminated" class="text-red-400">
            ⚔️ {{ game.stepData.eliminated }} 被投票出局
          </div>
          <div v-else class="text-gray-500">准备就绪</div>
          
          <div class="flex items-center gap-3">
            <button 
              v-if="!game.isOver"
              @click="$emit('step')"
              class="btn btn-primary flex items-center gap-2"
            >
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path>
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
              </svg>
              下一步
            </button>
            <button 
              v-if="game.isOver"
              @click="$emit('stop')"
              class="btn btn-secondary"
            >
              结束游戏
            </button>
          </div>
        </div>
      </div>
    </div>
    
    <!-- 右侧：游戏信息 -->
    <div class="space-y-6">
      <!-- 阵营统计 -->
      <div class="card p-6">
        <h3 class="text-lg font-semibold mb-4 flex items-center gap-2">
          <svg class="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path>
          </svg>
          阵营统计
        </h3>
        
        <div class="space-y-4">
          <!-- 狼人阵营 -->
          <div>
            <div class="flex items-center justify-between mb-2">
              <span class="flex items-center gap-2">
                <span class="text-2xl">🐺</span>
                <span class="text-red-400 font-medium">狼人阵营</span>
              </span>
              <span class="text-gray-400">{{ game.stats.wolvesAlive }}/{{ game.players.filter(p => p.team === 'evil').length }}</span>
            </div>
            <div class="progress-bar">
              <div 
                class="progress-fill bg-gradient-to-r from-red-600 to-red-500"
                :style="{ width: `${(game.stats.wolvesAlive / game.players.filter(p => p.team === 'evil').length) * 100}%` }"
              ></div>
            </div>
          </div>
          
          <!-- 好人阵营 -->
          <div>
            <div class="flex items-center justify-between mb-2">
              <span class="flex items-center gap-2">
                <span class="text-2xl">👨‍🌾</span>
                <span class="text-green-400 font-medium">好人阵营</span>
              </span>
              <span class="text-gray-400">{{ game.stats.goodsAlive }}/{{ game.players.filter(p => p.team === 'good').length }}</span>
            </div>
            <div class="progress-bar">
              <div 
                class="progress-fill bg-gradient-to-r from-green-600 to-green-500"
                :style="{ width: `${(game.stats.goodsAlive / game.players.filter(p => p.team === 'good').length) * 100}%` }"
              ></div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- 胜利条件 -->
      <div class="card p-6">
        <h3 class="text-lg font-semibold mb-4 flex items-center gap-2">
          <svg class="w-5 h-5 text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
          </svg>
          胜利条件
        </h3>
        
        <div class="space-y-3 text-sm">
          <div class="flex items-start gap-2 p-3 rounded-lg bg-red-900/30">
            <span class="text-red-400">🐺</span>
            <div>
              <div class="font-medium text-red-300">狼人胜利</div>
              <div class="text-gray-400">狼人数 ≥ 好人数</div>
            </div>
          </div>
          <div class="flex items-start gap-2 p-3 rounded-lg bg-green-900/30">
            <span class="text-green-400">🏆</span>
            <div>
              <div class="font-medium text-green-300">好人胜利</div>
              <div class="text-gray-400">消灭所有狼人</div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- 游戏结束 -->
      <div 
        v-if="game.isOver"
        :class="[
          'card p-6 text-center',
          game.winner === 'good' ? 'border-green-500/50 bg-green-900/20' : 'border-red-500/50 bg-red-900/20'
        ]"
      >
        <div class="text-5xl mb-4">{{ game.winner === 'good' ? '🎉' : '🐺' }}</div>
        <h3 class="text-2xl font-bold mb-2" :class="game.winner === 'good' ? 'text-green-400' : 'text-red-400'">
          {{ game.winner === 'good' ? '好人阵营胜利！' : '狼人阵营胜利！' }}
        </h3>
        <p class="text-gray-400">游戏结束</p>
      </div>
      
      <!-- 角色说明 -->
      <div class="card p-6">
        <h3 class="text-lg font-semibold mb-4 flex items-center gap-2">
          <svg class="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
          </svg>
          角色说明
        </h3>
        
        <div class="space-y-2 text-sm">
          <div class="flex items-center justify-between p-2 rounded-lg bg-gray-800/50">
            <span>🐺 狼人</span>
            <span class="text-gray-400">夜间刀人</span>
          </div>
          <div class="flex items-center justify-between p-2 rounded-lg bg-gray-800/50">
            <span>🔮 预言家</span>
            <span class="text-gray-400">查验身份</span>
          </div>
          <div class="flex items-center justify-between p-2 rounded-lg bg-gray-800/50">
            <span>🧙 女巫</span>
            <span class="text-gray-400">解药/毒药</span>
          </div>
          <div class="flex items-center justify-between p-2 rounded-lg bg-gray-800/50">
            <span>👨‍🌾 平民</span>
            <span class="text-gray-400">白天投票</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  game: {
    type: Object,
    required: true
  },
  isPlaying: {
    type: Boolean,
    default: false
  }
})

defineEmits(['step', 'stop'])

const getPhaseClass = (phase) => {
  const classes = {
    'night_wolf': 'phase-night',
    'night_seer': 'phase-night',
    'night_witch': 'phase-night',
    'night_result': 'phase-night',
    'day_start': 'phase-day',
    'speech': 'phase-speech',
    'vote': 'phase-vote',
    'day_end': 'phase-day'
  }
  return classes[phase] || 'phase-day'
}

const getRoleText = (role) => {
  const texts = {
    'werewolf': '狼人',
    'seer': '预言家',
    'witch': '女巫',
    'hunter': '猎人',
    'villager': '平民'
  }
  return texts[role] || role
}

const getRoleColor = (role) => {
  const colors = {
    'werewolf': 'text-red-400',
    'seer': 'text-blue-400',
    'witch': 'text-purple-400',
    'hunter': 'text-orange-400',
    'villager': 'text-green-400'
  }
  return colors[role] || 'text-gray-400'
}
</script>
