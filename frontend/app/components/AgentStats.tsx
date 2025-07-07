'use client'

import { useState } from 'react'
import { RefreshCw, Users, Activity, Clock, TrendingUp, Zap, CheckCircle, XCircle, Puzzle } from 'lucide-react'

interface SystemStatusProps {
  stats: any
  onRefresh: () => void
}

export default function SystemStatus({ stats, onRefresh }: SystemStatusProps) {
  const [isRefreshing, setIsRefreshing] = useState(false)

  const handleRefresh = async () => {
    setIsRefreshing(true)
    await onRefresh()
    setIsRefreshing(false)
  }

  if (!stats) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">System Status</h3>
          <button
            onClick={handleRefresh}
            className="p-2 text-gray-500 hover:text-gray-700"
            data-testid="refresh-stats"
          >
            <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`} />
          </button>
        </div>
        <div className="text-center text-gray-500 py-8">
          <Activity className="w-8 h-8 mx-auto mb-2 text-gray-300" />
          <p>Loading system status...</p>
        </div>
      </div>
    )
  }

  const {
    active_sessions = 0,
    available_plugins = 0,
    connector_status = {},
    reasoning_agent_status = "unknown",
    server_status = "unknown",
    websocket_connections = 0,
    last_updated = "Unknown"
  } = stats

  // Calculate connector health
  const connectorEntries = Object.entries(connector_status)
  const healthyConnectors = connectorEntries.filter(([_, status]) => status === true).length
  const totalConnectors = connectorEntries.length
  const connectorHealthRate = totalConnectors > 0 ? (healthyConnectors / totalConnectors) * 100 : 0

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900">System Status</h3>
        <button
          onClick={handleRefresh}
          className="p-2 text-gray-500 hover:text-gray-700"
          data-testid="refresh-stats"
        >
          <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* System Status Cards */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-blue-50 rounded-lg p-4">
          <div className="flex items-center">
            <Activity className="w-5 h-5 text-blue-500 mr-2" />
            <div>
              <div className="text-2xl font-bold text-blue-900">{active_sessions}</div>
              <div className="text-sm text-blue-600">Active Sessions</div>
            </div>
          </div>
        </div>

        <div className="bg-green-50 rounded-lg p-4">
          <div className="flex items-center">
            <Puzzle className="w-5 h-5 text-green-500 mr-2" />
            <div>
              <div className="text-2xl font-bold text-green-900">{available_plugins}</div>
              <div className="text-sm text-green-600">Available Plugins</div>
            </div>
          </div>
        </div>

        <div className="bg-yellow-50 rounded-lg p-4">
          <div className="flex items-center">
            <Users className="w-5 h-5 text-yellow-500 mr-2" />
            <div>
              <div className="text-2xl font-bold text-yellow-900">{websocket_connections}</div>
              <div className="text-sm text-yellow-600">WebSocket Connections</div>
            </div>
          </div>
        </div>

        <div className="bg-purple-50 rounded-lg p-4">
          <div className="flex items-center">
            <TrendingUp className="w-5 h-5 text-purple-500 mr-2" />
            <div>
              <div className="text-2xl font-bold text-purple-900">{connectorHealthRate.toFixed(1)}%</div>
              <div className="text-sm text-purple-600">Connector Health</div>
            </div>
          </div>
        </div>
      </div>

      {/* System Components Status */}
      <div className="mb-6">
        <h4 className="text-md font-medium text-gray-900 mb-3">System Components</h4>
        <div className="space-y-2">
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center">
              <Zap className="w-4 h-4 text-blue-500 mr-2" />
              <span className="text-sm font-medium text-gray-900">Reasoning Agent</span>
            </div>
            <div className="flex items-center">
              {reasoning_agent_status === 'active' ? (
                <CheckCircle className="w-4 h-4 text-green-500" />
              ) : (
                <XCircle className="w-4 h-4 text-red-500" />
              )}
              <span className="ml-2 text-sm text-gray-600 capitalize">{reasoning_agent_status}</span>
            </div>
          </div>

          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center">
              <Activity className="w-4 h-4 text-green-500 mr-2" />
              <span className="text-sm font-medium text-gray-900">Server</span>
            </div>
            <div className="flex items-center">
              {server_status === 'active' ? (
                <CheckCircle className="w-4 h-4 text-green-500" />
              ) : (
                <XCircle className="w-4 h-4 text-red-500" />
              )}
              <span className="ml-2 text-sm text-gray-600 capitalize">{server_status}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Connector Status */}
      {connectorEntries.length > 0 && (
        <div className="mb-6">
          <h4 className="text-md font-medium text-gray-900 mb-3">Connector Status</h4>
          <div className="space-y-2">
            {connectorEntries.map(([name, status]) => (
              <div key={name} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center">
                  <Clock className="w-4 h-4 text-blue-500 mr-2" />
                  <span className="text-sm font-medium text-gray-900 capitalize">
                    {name.replace(/_/g, ' ')}
                  </span>
                </div>
                <div className="flex items-center">
                  {status ? (
                    <CheckCircle className="w-4 h-4 text-green-500" />
                  ) : (
                    <XCircle className="w-4 h-4 text-red-500" />
                  )}
                  <span className="ml-2 text-sm text-gray-600">
                    {status ? 'Connected' : 'Disconnected'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Last Updated */}
      <div className="mt-4 text-xs text-gray-500 text-center">
        Last updated: {last_updated}
      </div>
    </div>
  )
}
