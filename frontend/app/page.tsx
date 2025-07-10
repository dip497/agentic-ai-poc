'use client'

import { useState, useEffect } from 'react'
import ChatInterface from './components/ChatInterface'
import SystemStatus from './components/AgentStats'
import AgentStudio from './components/AgentStudio'
import { AGUIClient } from './lib/agui-client'
import { Button } from './components/ui/button'
import { MessageSquare, Settings, Code } from 'lucide-react'

export default function Home() {
  const [client, setClient] = useState<AGUIClient | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [currentSession, setCurrentSession] = useState<string>('')
  const [systemStatus, setSystemStatus] = useState<any>(null)
  const [currentView, setCurrentView] = useState<'chat' | 'studio'>('chat')

  useEffect(() => {
    // Initialize AG-UI client
    const aguiClient = new AGUIClient('ws://localhost:8081')
    
    aguiClient.onConnect = () => {
      setIsConnected(true)
      console.log('Connected to AG-UI server')
    }
    
    aguiClient.onDisconnect = () => {
      setIsConnected(false)
      console.log('Disconnected from AG-UI server')
    }
    
    // Reasoning steps are now handled in ChatInterface
    
    aguiClient.connect()
    setClient(aguiClient)
    
    // Generate proper UUID for session ID
    const generateUUID = () => {
      return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
      });
    };
    setCurrentSession(generateUUID())
    
    // Fetch system status
    fetchSystemStatus()
    
    return () => {
      aguiClient.disconnect()
    }
  }, [])

  const fetchSystemStatus = async () => {
    try {
      const response = await fetch('http://localhost:8081/api/system/status')
      const status = await response.json()
      setSystemStatus(status)
    } catch (error) {
      console.error('Failed to fetch system status:', error)
    }
  }



  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Moveworks AI Platform
              </h1>
              <p className="text-sm text-gray-600">
                LangGraph reasoning, human-in-the-loop, and Agent Studio
              </p>
            </div>

            <div className="flex items-center space-x-4">
              <div className={`flex items-center space-x-2 ${isConnected ? 'text-green-600' : 'text-red-600'}`}>
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
                <span className="text-sm font-medium">
                  {isConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
              <div className="text-sm text-gray-500">
                Session: {currentSession}
              </div>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="mt-4 flex space-x-1">
            <Button
              variant={currentView === 'chat' ? 'default' : 'ghost'}
              onClick={() => setCurrentView('chat')}
              className="flex items-center space-x-2"
            >
              <MessageSquare className="w-4 h-4" />
              <span>Chat Interface</span>
            </Button>
            <Button
              variant={currentView === 'studio' ? 'default' : 'ghost'}
              onClick={() => setCurrentView('studio')}
              className="flex items-center space-x-2"
            >
              <Code className="w-4 h-4" />
              <span>Agent Studio</span>
            </Button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-6">
        {currentView === 'chat' ? (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Chat Interface */}
            <div className="lg:col-span-3">
              <ChatInterface
                client={client}
                sessionId={currentSession}
                isConnected={isConnected}
              />
            </div>

            {/* Side Panel */}
            <div className="space-y-6">
              {/* System Status */}
              <SystemStatus
                stats={systemStatus}
                onRefresh={fetchSystemStatus}
              />
            </div>
          </div>
        ) : (
          <AgentStudio />
        )}
      </div>
    </div>
  )
}
