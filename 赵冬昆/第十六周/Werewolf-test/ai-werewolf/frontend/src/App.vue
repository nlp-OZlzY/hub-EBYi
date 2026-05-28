<template>
  <div class="min-h-screen">
    <Header @start-game="startGame" @show-leaderboard="showLeaderboard = true" />
    
    <div class="container mx-auto px-4 py-6">
      <GameBoard 
        v-if="currentGame" 
        :game="currentGame" 
        :is-playing="isPlaying"
        @step="stepGame"
        @stop="stopGame"
      />
      
      <div v-else class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div class="lg:col-span-2">
          <div class="card p-8 text-center">
            <div class="mb-6">
              <div class="w-24 h-24 mx-auto mb-4 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                <svg class="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
              </div>
              <h2 class="text-2xl font-bold mb-2">AI狼人杀</h2>
              <p class="text-gray-400">多智能体博弈系统</p>
            </div>
            
            <div class="space-y-4">
              <button @click="startGame" class="btn btn-primary w-full py-3 text-lg">
                🎮 开始新游戏
              </button>
              <button @click="loadDemoGame" class="btn btn-secondary w-full py-3 text-lg">
                📊 加载演示数据
              </button>
            </div>
            
            <div class="mt-8 grid grid-cols-3 gap-4">
              <div class="bg-gray-800/50 rounded-lg p-4">
                <div class="text-2xl font-bold text-indigo-400">6</div>
                <div class="text-sm text-gray-400">玩家数</div>
              </div>
              <div class="bg-gray-800/50 rounded-lg p-4">
                <div class="text-2xl font-bold text-red-400">2</div>
                <div class="text-sm text-gray-400">狼人</div>
              </div>
              <div class="bg-gray-800/50 rounded-lg p-4">
                <div class="text-2xl font-bold text-green-400">4</div>
                <div class="text-sm text-gray-400">好人</div>
              </div>
            </div>
          </div>
        </div>
        
        <div class="space-y-6">
          <div class="card p-6">
            <h3 class="text-lg font-semibold mb-4 flex items-center gap-2">
              <svg class="w-5 h-5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
              </svg>
              游戏规则
            </h3>
            <ul class="space-y-3 text-sm text-gray-400">
              <li class="flex items-start gap-2">
                <span class="text-indigo-400">•</span>
                <span>狼人每晚刀杀一名玩家</span>
              </li>
              <li class="flex items-start gap-2">
                <span class="text-indigo-400">•</span>
                <span>预言家每晚查验身份</span>
              </li>
              <li class="flex items-start gap-2">
                <span class="text-indigo-400">•</span>
                <span>女巫拥有解药和毒药</span>
              </li>
              <li class="flex items-start gap-2">
                <span class="text-indigo-400">•</span>
                <span>白天投票出局嫌疑最大者</span>
              </li>
            </ul>
          </div>
          
          <div class="card p-6">
            <h3 class="text-lg font-semibold mb-4 flex items-center gap-2">
              <svg class="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path>
              </svg>
              阵营说明
            </h3>
            <div class="space-y-2">
              <div class="flex items-center justify-between p-2 rounded-lg bg-red-900/30">
                <span class="text-red-400">狼人阵营</span>
                <span class="text-sm text-gray-400">2人</span>
              </div>
              <div class="flex items-center justify-between p-2 rounded-lg bg-green-900/30">
                <span class="text-green-400">好人阵营</span>
                <span class="text-sm text-gray-400">4人</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <Leaderboard 
      v-if="showLeaderboard" 
      @close="showLeaderboard = false" 
    />
  </div>
</template>

<script setup>
import { ref } from 'vue'
import Header from './components/Header.vue'
import GameBoard from './components/GameBoard.vue'
import Leaderboard from './components/Leaderboard.vue'

const currentGame = ref(null)
const isPlaying = ref(false)
const showLeaderboard = ref(false)

const startGame = () => {
  // 创建游戏
  currentGame.value = {
    gameId: 'demo-001',
    day: 1,
    phase: 'night_wolf',
    phaseText: '狼人行动',
    isOver: false,
    winner: null,
    players: [
      { id: 0, name: '玩家1', role: 'werewolf', isAlive: true, team: 'evil', avatar: '🐺' },
      { id: 1, name: '玩家2', role: 'seer', isAlive: true, team: 'good', avatar: '🔮' },
      { id: 2, name: '玩家3', role: 'witch', isAlive: true, team: 'good', avatar: '🧙' },
      { id: 3, name: '玩家4', role: 'villager', isAlive: true, team: 'good', avatar: '👨‍🌾' },
      { id: 4, name: '玩家5', role: 'werewolf', isAlive: true, team: 'evil', avatar: '🐺' },
      { id: 5, name: '玩家6', role: 'villager', isAlive: true, team: 'good', avatar: '👨‍🌾' },
    ],
    dialogues: [],
    stepData: {},
    stats: {
      wolvesAlive: 2,
      goodsAlive: 4,
    }
  }
  isPlaying.value = true
}

const loadDemoGame = () => {
  // 加载演示数据
  currentGame.value = {
    gameId: 'demo-002',
    day: 3,
    phase: 'speech',
    phaseText: '发言阶段',
    isOver: false,
    winner: null,
    players: [
      { id: 0, name: '玩家1', role: 'werewolf', isAlive: false, team: 'evil', avatar: '🐺' },
      { id: 1, name: '玩家2', role: 'seer', isAlive: true, team: 'good', avatar: '🔮' },
      { id: 2, name: '玩家3', role: 'witch', isAlive: true, team: 'good', avatar: '🧙' },
      { id: 3, name: '玩家4', role: 'villager', isAlive: true, team: 'good', avatar: '👨‍🌾' },
      { id: 4, name: '玩家5', role: 'werewolf', isAlive: true, team: 'evil', avatar: '🐺' },
      { id: 5, name: '玩家6', role: 'villager', isAlive: false, team: 'good', avatar: '👨‍🌾' },
    ],
    dialogues: [
      { speaker: '玩家2', content: '昨晚查验了玩家1，是狼人！', phase: 'speech', time: '第2天' },
      { speaker: '玩家4', content: '我相信预言家，建议出玩家1', phase: 'speech', time: '第2天' },
      { speaker: '玩家5', content: '我是好人，昨天投错了', phase: 'speech', time: '第2天' },
    ],
    stepData: {},
    stats: {
      wolvesAlive: 1,
      goodsAlive: 3,
    }
  }
  isPlaying.value = false
}

const stepGame = () => {
  if (!currentGame.value || currentGame.value.isOver) return
  
  const phases = ['night_wolf', 'night_seer', 'night_witch', 'night_result', 
                  'day_start', 'speech', 'vote', 'day_end']
  const phaseTexts = {
    'night_wolf': '狼人行动',
    'night_seer': '预言家查验',
    'night_witch': '女巫用药',
    'night_result': '夜晚结果',
    'day_start': '白天开始',
    'speech': '发言阶段',
    'vote': '投票阶段',
    'day_end': '白天结束'
  }
  
  const currentIndex = phases.indexOf(currentGame.value.phase)
  let nextIndex = (currentIndex + 1) % phases.length
  
  if (nextIndex === 0) {
    currentGame.value.day += 1
  }
  
  currentGame.value.phase = phases[nextIndex]
  currentGame.value.phaseText = phaseTexts[phases[nextIndex]]
  
  // 模拟游戏事件
  if (currentGame.value.phase === 'speech') {
    const alivePlayers = currentGame.value.players.filter(p => p.isAlive)
    const speaker = alivePlayers[Math.floor(Math.random() * alivePlayers.length)]
    const speeches = [
      '我是好人，昨晚没有任何信息',
      '我怀疑玩家5的发言，他可能是狼人',
      '预言家请跳出来报查验',
      '我是平民，跟着预言家走',
      '昨晚我被刀了，女巫救了我',
    ]
    currentGame.value.dialogues.push({
      speaker: speaker.name,
      content: speeches[Math.floor(Math.random() * speeches.length)],
      phase: 'speech',
      time: `第${currentGame.value.day}天`
    })
  }
  
  if (currentGame.value.phase === 'vote') {
    const alivePlayers = currentGame.value.players.filter(p => p.isAlive)
    if (alivePlayers.length > 0) {
      const voted = alivePlayers[Math.floor(Math.random() * alivePlayers.length)]
      currentGame.value.stepData = { eliminated: voted.name }
      voted.isAlive = false
      
      // 更新统计
      if (voted.team === 'evil') {
        currentGame.value.stats.wolvesAlive -= 1
      } else {
        currentGame.value.stats.goodsAlive -= 1
      }
    }
  }
  
  // 检查胜负
  if (currentGame.value.stats.wolvesAlive <= 0) {
    currentGame.value.isOver = true
    currentGame.value.winner = 'good'
    currentGame.value.phaseText = '游戏结束 - 好人胜利'
  } else if (currentGame.value.stats.wolvesAlive >= currentGame.value.stats.goodsAlive) {
    currentGame.value.isOver = true
    currentGame.value.winner = 'evil'
    currentGame.value.phaseText = '游戏结束 - 狼人胜利'
  }
}

const stopGame = () => {
  isPlaying.value = false
}
</script>
