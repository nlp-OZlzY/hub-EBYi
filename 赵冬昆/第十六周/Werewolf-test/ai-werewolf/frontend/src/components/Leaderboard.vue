<template>
  <div class="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
    <div class="card w-full max-w-4xl max-h-[80vh] overflow-hidden">
      <!-- 头部 -->
      <div class="flex items-center justify-between p-6 border-b border-gray-800">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-yellow-500 to-orange-500 flex items-center justify-center">
            <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z"></path>
            </svg>
          </div>
          <div>
            <h2 class="text-xl font-bold">排行榜</h2>
            <p class="text-sm text-gray-400">Agent 竞技排名</p>
          </div>
        </div>
        <button @click="$emit('close')" class="p-2 rounded-lg hover:bg-gray-800 transition-colors">
          <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
          </svg>
        </button>
      </div>
      
      <!-- 内容 -->
      <div class="p-6 overflow-y-auto max-h-[60vh]">
        <!-- 总排行榜 -->
        <div class="mb-8">
          <h3 class="text-lg font-semibold mb-4 flex items-center gap-2">
            <span class="text-2xl">🏆</span>
            总排行榜
          </h3>
          
          <div class="space-y-2">
            <div 
              v-for="(entry, index) in leaderboardData" 
              :key="entry.agentId"
              :class="[
                'flex items-center gap-4 p-4 rounded-xl transition-colors',
                index === 0 ? 'bg-gradient-to-r from-yellow-900/50 to-orange-900/50 border border-yellow-500/30' :
                index === 1 ? 'bg-gradient-to-r from-gray-700/50 to-gray-600/50 border border-gray-400/30' :
                index === 2 ? 'bg-gradient-to-r from-orange-900/30 to-yellow-900/30 border border-orange-600/30' :
                'bg-gray-800/50 hover:bg-gray-700/50'
              ]"
            >
              <!-- 排名 -->
              <div 
                :class="[
                  'w-10 h-10 rounded-full flex items-center justify-center font-bold text-lg',
                  index === 0 ? 'bg-yellow-500 text-black' :
                  index === 1 ? 'bg-gray-400 text-black' :
                  index === 2 ? 'bg-orange-500 text-black' :
                  'bg-gray-700 text-gray-300'
                ]"
              >
                {{ index + 1 }}
              </div>
              
              <!-- Agent 信息 -->
              <div class="flex-1">
                <div class="flex items-center gap-3">
                  <span class="font-medium">{{ entry.agentId }}</span>
                  <span class="px-2 py-0.5 rounded-full text-xs" :class="getModelClass(entry.model)">
                    {{ entry.model }}
                  </span>
                  <span 
                    v-if="entry.trend"
                    :class="[
                      'flex items-center gap-1 text-xs',
                      entry.trend === 'up' ? 'text-green-400' : 
                      entry.trend === 'down' ? 'text-red-400' : 'text-gray-400'
                    ]"
                  >
                    <svg v-if="entry.trend === 'up'" class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 10l7-7m0 0l7 7m-7-7v18"></path>
                    </svg>
                    <svg v-else-if="entry.trend === 'down'" class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3"></path>
                    </svg>
                    <svg v-else class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"></path>
                    </svg>
                    {{ entry.trend === 'up' ? '上升' : entry.trend === 'down' ? '下降' : '稳定' }}
                  </span>
                </div>
              </div>
              
              <!-- 统计数据 -->
              <div class="grid grid-cols-4 gap-6 text-center">
                <div>
                  <div class="text-lg font-bold text-indigo-400">{{ entry.score }}</div>
                  <div class="text-xs text-gray-400">均分</div>
                </div>
                <div>
                  <div class="text-lg font-bold text-green-400">{{ entry.winRate }}%</div>
                  <div class="text-xs text-gray-400">胜率</div>
                </div>
                <div>
                  <div class="text-lg font-bold text-yellow-400">{{ entry.mvpRate }}%</div>
                  <div class="text-xs text-gray-400">MVP率</div>
                </div>
                <div>
                  <div class="text-lg font-bold text-gray-300">{{ entry.games }}</div>
                  <div class="text-xs text-gray-400">场次</div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <!-- 按角色排行榜 -->
        <div>
          <h3 class="text-lg font-semibold mb-4 flex items-center gap-2">
            <span class="text-2xl">🎭</span>
            按角色排行榜
          </h3>
          
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div 
              v-for="role in roleLeaderboards" 
              :key="role.name"
              class="bg-gray-800/50 rounded-xl p-4"
            >
              <div class="flex items-center gap-2 mb-3">
                <span class="text-xl">{{ role.icon }}</span>
                <span class="font-medium" :class="role.color">{{ role.name }}</span>
              </div>
              
              <div class="space-y-2">
                <div 
                  v-for="(entry, index) in role.entries.slice(0, 3)" 
                  :key="index"
                  class="flex items-center justify-between text-sm"
                >
                  <div class="flex items-center gap-2">
                    <span class="text-gray-500">{{ index + 1 }}.</span>
                    <span>{{ entry.agentId }}</span>
                  </div>
                  <span class="text-indigo-400">{{ entry.score }}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
defineEmits(['close'])

const leaderboardData = [
  { agentId: 'Agent Alpha', model: 'Qwen 3.5', score: 142.5, winRate: 78, mvpRate: 45, games: 23, trend: 'up' },
  { agentId: 'Agent Beta', model: 'GPT-4', score: 138.2, winRate: 72, mvpRate: 38, games: 28, trend: 'up' },
  { agentId: 'Agent Gamma', model: 'Claude 3', score: 135.8, winRate: 68, mvpRate: 35, games: 25, trend: 'stable' },
  { agentId: 'Agent Delta', model: 'Qwen 3.5', score: 128.4, winRate: 65, mvpRate: 30, games: 21, trend: 'down' },
  { agentId: 'Agent Epsilon', model: 'GPT-4', score: 122.1, winRate: 60, mvpRate: 25, games: 19, trend: 'stable' },
  { agentId: 'Agent Zeta', model: 'Claude 3', score: 118.7, winRate: 58, mvpRate: 22, games: 17, trend: 'up' },
]

const roleLeaderboards = [
  {
    name: '狼人',
    icon: '🐺',
    color: 'text-red-400',
    entries: [
      { agentId: 'Agent Alpha', score: 156.2 },
      { agentId: 'Agent Beta', score: 148.5 },
      { agentId: 'Agent Gamma', score: 142.1 },
    ]
  },
  {
    name: '预言家',
    icon: '🔮',
    color: 'text-blue-400',
    entries: [
      { agentId: 'Agent Beta', score: 138.8 },
      { agentId: 'Agent Alpha', score: 135.2 },
      { agentId: 'Agent Delta', score: 128.4 },
    ]
  },
  {
    name: '女巫',
    icon: '🧙',
    color: 'text-purple-400',
    entries: [
      { agentId: 'Agent Gamma', score: 140.6 },
      { agentId: 'Agent Alpha', score: 136.8 },
      { agentId: 'Agent Epsilon', score: 124.3 },
    ]
  },
  {
    name: '平民',
    icon: '👨‍🌾',
    color: 'text-green-400',
    entries: [
      { agentId: 'Agent Alpha', score: 128.9 },
      { agentId: 'Agent Zeta', score: 122.5 },
      { agentId: 'Agent Beta', score: 118.7 },
    ]
  },
]

const getModelClass = (model) => {
  if (model.includes('Qwen')) return 'bg-indigo-900/50 text-indigo-300'
  if (model.includes('GPT')) return 'bg-green-900/50 text-green-300'
  if (model.includes('Claude')) return 'bg-orange-900/50 text-orange-300'
  return 'bg-gray-700 text-gray-300'
}
</script>
