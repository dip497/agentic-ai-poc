'use client'

import { useState } from 'react'
import { Brain, Trash2, ChevronDown, ChevronRight, Clock, Zap, AlertTriangle, CheckCircle } from 'lucide-react'

interface ReasoningTraceProps {
  trace: any[]
  onClear: () => void
}

export default function ReasoningTrace({ trace, onClear }: ReasoningTraceProps) {
  const [isExpanded, setIsExpanded] = useState(true)
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set())

  const toggleStepExpansion = (index: number) => {
    const newExpanded = new Set(expandedSteps)
    if (newExpanded.has(index)) {
      newExpanded.delete(index)
    } else {
      newExpanded.add(index)
    }
    setExpandedSteps(newExpanded)
  }

  const getStepIcon = (step: any) => {
    const stepType = step.step || step.state || 'unknown'
    
    switch (stepType) {
      case 'process_matching':
      case 'process_matching_started':
        return <Zap className="w-4 h-4 text-blue-500" />
      case 'slot_inference':
      case 'slot_clarification':
        return <AlertTriangle className="w-4 h-4 text-yellow-500" />
      case 'activity_execution':
        return <Brain className="w-4 h-4 text-green-500" />
      case 'decision_evaluation':
        return <CheckCircle className="w-4 h-4 text-purple-500" />
      case 'response_generation':
        return <CheckCircle className="w-4 h-4 text-green-600" />
      case 'error':
        return <AlertTriangle className="w-4 h-4 text-red-500" />
      default:
        return <Clock className="w-4 h-4 text-gray-500" />
    }
  }

  const getStepColor = (step: any) => {
    const stepType = step.step || step.state || 'unknown'
    
    switch (stepType) {
      case 'process_matching':
      case 'process_matching_started':
        return 'border-blue-200 bg-blue-50'
      case 'slot_inference':
      case 'slot_clarification':
        return 'border-yellow-200 bg-yellow-50'
      case 'activity_execution':
        return 'border-green-200 bg-green-50'
      case 'decision_evaluation':
        return 'border-purple-200 bg-purple-50'
      case 'response_generation':
        return 'border-green-200 bg-green-50'
      case 'error':
        return 'border-red-200 bg-red-50'
      default:
        return 'border-gray-200 bg-gray-50'
    }
  }

  const formatStepTitle = (step: any) => {
    const stepType = step.step || step.state || 'unknown'
    return stepType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
  }

  const formatTimestamp = (timestamp: string) => {
    try {
      return new Date(timestamp).toLocaleTimeString()
    } catch {
      return timestamp
    }
  }

  return (
    <div className="bg-white rounded-lg shadow-lg">
      {/* Header */}
      <div className="border-b p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="p-1 hover:bg-gray-100 rounded"
            >
              {isExpanded ? (
                <ChevronDown className="w-4 h-4 text-gray-500" />
              ) : (
                <ChevronRight className="w-4 h-4 text-gray-500" />
              )}
            </button>
            <Brain className="w-5 h-5 text-purple-500" />
            <h3 className="text-lg font-semibold text-gray-900">Reasoning Trace</h3>
            <span className="bg-gray-100 text-gray-600 text-xs px-2 py-1 rounded-full">
              {trace.length} steps
            </span>
          </div>
          {trace.length > 0 && (
            <button
              onClick={onClear}
              className="p-2 text-gray-500 hover:text-red-600 hover:bg-red-50 rounded"
              title="Clear trace"
              data-testid="clear-trace"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Content */}
      {isExpanded && (
        <div className="p-4">
          {trace.length === 0 ? (
            <div className="text-center text-gray-500 py-8">
              <Brain className="w-8 h-8 mx-auto mb-2 text-gray-300" />
              <p>No reasoning steps yet</p>
              <p className="text-sm mt-1">Start a conversation to see the reasoning process</p>
            </div>
          ) : (
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {trace.map((step, index) => (
                <div
                  key={index}
                  className={`border rounded-lg p-3 ${getStepColor(step)}`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start space-x-2 flex-1">
                      {getStepIcon(step)}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                          <h4 className="text-sm font-medium text-gray-900">
                            {formatStepTitle(step)}
                          </h4>
                          {Object.keys(step).length > 3 && (
                            <button
                              onClick={() => toggleStepExpansion(index)}
                              className="p-1 hover:bg-white hover:bg-opacity-50 rounded"
                              data-testid={`expand-step-${index}`}
                            >
                              {expandedSteps.has(index) ? (
                                <ChevronDown className="w-3 h-3 text-gray-500" />
                              ) : (
                                <ChevronRight className="w-3 h-3 text-gray-500" />
                              )}
                            </button>
                          )}
                        </div>
                        
                        {/* Basic info always shown */}
                        <div className="text-xs text-gray-600 mt-1">
                          {step.timestamp && (
                            <span>{formatTimestamp(step.timestamp)}</span>
                          )}
                          {step.action && (
                            <span className="ml-2">• {step.action}</span>
                          )}
                        </div>

                        {/* Expanded details */}
                        {expandedSteps.has(index) && (
                          <div className="mt-2 space-y-2">
                            {Object.entries(step).map(([key, value]) => {
                              if (['step', 'state', 'timestamp', 'action'].includes(key)) {
                                return null
                              }
                              
                              return (
                                <div key={key} className="text-xs">
                                  <span className="font-medium text-gray-700">
                                    {key.replace(/_/g, ' ')}:
                                  </span>
                                  <span className="ml-2 text-gray-600">
                                    {typeof value === 'object' 
                                      ? JSON.stringify(value, null, 2)
                                      : String(value)
                                    }
                                  </span>
                                </div>
                              )
                            })}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
