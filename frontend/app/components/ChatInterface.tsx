'use client'

import { useState, useRef, useEffect } from 'react'
import { flushSync } from 'react-dom'
import { Send, Bot, User, AlertCircle, CheckCircle, XCircle } from 'lucide-react'
import { AGUIClient, ConfirmationRequest, SlotClarificationRequest } from '../lib/agui-client'

interface Message {
  id: string
  type: 'user' | 'assistant' | 'system' | 'confirmation' | 'slot_clarification'
  content: string
  timestamp: Date
  metadata?: any
}

interface ChatInterfaceProps {
  client: AGUIClient | null
  sessionId: string
  isConnected: boolean
}

export default function ChatInterface({ client, sessionId, isConnected }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [currentAgent, setCurrentAgent] = useState<any>(null)
  const [pendingConfirmation, setPendingConfirmation] = useState<ConfirmationRequest | null>(null)
  const [pendingSlotClarification, setPendingSlotClarification] = useState<SlotClarificationRequest | null>(null)
  const [streamingMessage, setStreamingMessage] = useState('')
  const streamingMessageRef = useRef('')
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (client) {
      // Set up event handlers
      client.onTextMessageStart = (data) => {
        console.log('Message started:', data)
        // Clear any previous streaming message
        setStreamingMessage('')
        streamingMessageRef.current = ''
      }

      client.onMessage = (data) => {
        // Handle TEXT_MESSAGE_CONTENT events with delta field
        if (data.delta) {
          streamingMessageRef.current += data.delta
          // Force immediate React update to show streaming
          flushSync(() => {
            setStreamingMessage(streamingMessageRef.current)
          })
        }
      }

      client.onTextMessageEnd = (data) => {
        console.log('Message ended:', data)
        // Complete the streaming message
        if (streamingMessageRef.current) {
          addMessage('assistant', streamingMessageRef.current)
          setStreamingMessage('')
          streamingMessageRef.current = ''
          setIsLoading(false)
        }
      }

      client.onAgentSelected = (agentData) => {
        setCurrentAgent(agentData)
        addMessage('system', `🤖 Routed to ${agentData.routing_info?.selected_agent?.name || 'agent'} (confidence: ${(agentData.confidence * 100).toFixed(1)}%)`)
      }

      client.onReasoningStep = (step) => {
        // Add reasoning steps as system messages in the chat
        const stepName = step.step || step.state || 'reasoning'
        const action = step.action || 'Processing...'
        addMessage('system', `🧠 ${stepName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}: ${action}`)
      }

      client.onConfirmationRequired = (confirmation) => {
        setPendingConfirmation(confirmation)
        addMessage('confirmation', confirmation.action, { confirmation })
      }

      client.onSlotClarificationRequired = (clarification) => {
        setPendingSlotClarification(clarification)
        addMessage('slot_clarification', clarification.message, { clarification })
      }

      client.onError = (error) => {
        addMessage('system', `Error: ${error}`)
        setIsLoading(false)
      }

      // Initialize session
      client.initializeSession(sessionId)
    }
  }, [client, sessionId])

  useEffect(() => {
    scrollToBottom()
  }, [messages, streamingMessage])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const addMessage = (type: Message['type'], content: string, metadata?: any) => {
    const message: Message = {
      id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type,
      content,
      timestamp: new Date(),
      metadata
    }
    setMessages(prev => [...prev, message])
  }

  const handleSendMessage = async () => {
    if (!inputValue.trim() || !client || !isConnected) return

    const userMessage = inputValue.trim()
    setInputValue('')
    setIsLoading(true)
    setStreamingMessage('')

    // Add user message to chat
    addMessage('user', userMessage)

    // Clear any pending confirmations/clarifications
    setPendingConfirmation(null)
    setPendingSlotClarification(null)

    // Send message via WebSocket
    client.sendMessage({
      content: userMessage,
      user_id: 'user',
      session_id: sessionId,
      user_attributes: {
        interface: 'test_ui',
        timestamp: new Date().toISOString()
      }
    })
  }

  const handleConfirmation = (confirmed: boolean) => {
    if (!client || !pendingConfirmation) return

    client.sendConfirmationResponse(confirmed, sessionId)
    setPendingConfirmation(null)
    setIsLoading(true)
  }

  const handleSlotClarification = (value: string) => {
    if (!client || !pendingSlotClarification) return

    client.sendSlotClarification(value, sessionId)
    setPendingSlotClarification(null)
    setIsLoading(true)
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const renderMessage = (message: Message) => {
    const isUser = message.type === 'user'
    const isSystem = message.type === 'system'
    const isConfirmation = message.type === 'confirmation'
    const isSlotClarification = message.type === 'slot_clarification'

    return (
      <div key={message.id} className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
        <div className={`flex max-w-[80%] ${isUser ? 'flex-row-reverse' : 'flex-row'} items-start space-x-2`}>
          {/* Avatar */}
          <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
            isUser ? 'bg-blue-500 ml-2' : isSystem ? 'bg-gray-500 mr-2' : 'bg-green-500 mr-2'
          }`}>
            {isUser ? (
              <User className="w-4 h-4 text-white" />
            ) : isSystem ? (
              <AlertCircle className="w-4 h-4 text-white" />
            ) : (
              <Bot className="w-4 h-4 text-white" />
            )}
          </div>

          {/* Message Content */}
          <div className={`rounded-lg px-4 py-2 ${
            isUser 
              ? 'bg-blue-500 text-white' 
              : isSystem 
                ? 'bg-gray-100 text-gray-800 border'
                : isConfirmation
                  ? 'bg-yellow-50 text-yellow-800 border border-yellow-200'
                  : isSlotClarification
                    ? 'bg-blue-50 text-blue-800 border border-blue-200'
                    : 'bg-white text-gray-800 border'
          }`}>
            <div className="text-sm">{message.content}</div>
            
            {/* Confirmation Buttons */}
            {isConfirmation && pendingConfirmation && (
              <div className="mt-3 flex space-x-2">
                <button
                  onClick={() => handleConfirmation(true)}
                  className="flex items-center space-x-1 px-3 py-1 bg-green-500 text-white rounded text-xs hover:bg-green-600"
                  data-testid="confirm-yes"
                >
                  <CheckCircle className="w-3 h-3" />
                  <span>Yes, proceed</span>
                </button>
                <button
                  onClick={() => handleConfirmation(false)}
                  className="flex items-center space-x-1 px-3 py-1 bg-red-500 text-white rounded text-xs hover:bg-red-600"
                  data-testid="confirm-no"
                >
                  <XCircle className="w-3 h-3" />
                  <span>No, cancel</span>
                </button>
              </div>
            )}

            {/* Slot Clarification Input */}
            {isSlotClarification && pendingSlotClarification && (
              <div className="mt-3">
                <input
                  type="text"
                  placeholder={`Enter ${pendingSlotClarification.slot_name}...`}
                  className="w-full px-2 py-1 border rounded text-xs"
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      handleSlotClarification((e.target as HTMLInputElement).value)
                    }
                  }}
                  data-testid="slot-clarification-input"
                />
                {pendingSlotClarification.options && pendingSlotClarification.options.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {pendingSlotClarification.options.map((option, index) => (
                      <button
                        key={index}
                        onClick={() => handleSlotClarification(option)}
                        className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs hover:bg-blue-200"
                        data-testid={`slot-option-${index}`}
                      >
                        {option}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}

            <div className="text-xs opacity-70 mt-1">
              {message.timestamp.toLocaleTimeString()}
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg shadow-lg h-[600px] flex flex-col">
      {/* Header */}
      <div className="border-b p-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900">Chat Interface</h2>
          {currentAgent && (
            <div className="text-sm text-gray-600">
              Agent: {currentAgent.routing_info?.selected_agent?.name || 'Unknown'}
            </div>
          )}
        </div>
        {!isConnected && (
          <div className="mt-2 text-sm text-red-600">
            Not connected to server
          </div>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 mt-8">
            <Bot className="w-12 h-12 mx-auto mb-4 text-gray-300" />
            <p>Start a conversation with the Moveworks AI system</p>
            <p className="text-sm mt-2">Try asking about IT support, HR requests, or general questions</p>
          </div>
        )}
        
        {messages.map(renderMessage)}
        
        {/* Streaming message */}
        {streamingMessage && (
          <div className="flex justify-start mb-4">
            <div className="flex max-w-[80%] items-start space-x-2">
              <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center bg-green-500 mr-2">
                <Bot className="w-4 h-4 text-white" />
              </div>
              <div className="bg-white text-gray-800 border rounded-lg px-4 py-2">
                <div className="text-sm">{streamingMessage}<span className="animate-pulse">|</span></div>
              </div>
            </div>
          </div>
        )}
        
        {isLoading && !streamingMessage && (
          <div className="flex justify-start mb-4">
            <div className="flex items-center space-x-2 text-gray-500">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-500"></div>
              <span className="text-sm">Processing...</span>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t p-4">
        <div className="flex space-x-2">
          <input
            ref={inputRef}
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            className="flex-1 border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={!isConnected || isLoading}
            data-testid="chat-input"
          />
          <button
            onClick={handleSendMessage}
            disabled={!isConnected || isLoading || !inputValue.trim()}
            className="bg-blue-500 text-white rounded-lg px-4 py-2 hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
            data-testid="send-button"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  )
}
