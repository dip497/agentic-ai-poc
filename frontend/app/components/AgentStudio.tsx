import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from './ui/dialog';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import ProcessEditor from './ProcessEditor';
import {
  Plus,
  Play,
  Settings,
  Database,
  Code,
  TestTube,
  Rocket,
  Edit,
  Trash2,
  Eye
} from 'lucide-react';

interface ConversationalProcess {
  id: string;
  name: string;
  description: string;
  version: string;
  status: 'draft' | 'testing' | 'published' | 'archived';
  triggers: string[];
  keywords: string[];
  activities_count: number;
  slots_count: number;
  required_connectors: string[];
  created_at: string;
  updated_at: string;
  created_by: string;
  // Extended properties for full Moveworks-style configuration
  activities?: any[];
  slots?: any[];
  permissions?: any;
}

interface Connector {
  id: string;
  name: string;
  description: string;
  type: string;
  base_url: string;
  auth_type: string;
  status: 'active' | 'inactive' | 'testing';
  actions_count: number;
  created_at: string;
  updated_at: string;
}

interface TestResult {
  test_input: string;
  success: boolean;
  response: string;
  execution_time: number;
  steps_executed: any[];
  errors: string[];
  timestamp: string;
}

const AgentStudio: React.FC = () => {
  const [processes, setProcesses] = useState<ConversationalProcess[]>([]);
  const [connectors, setConnectors] = useState<Connector[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedProcess, setSelectedProcess] = useState<ConversationalProcess | null>(null);
  const [testResults, setTestResults] = useState<TestResult[]>([]);
  const [testInput, setTestInput] = useState('');
  const [testing, setTesting] = useState(false);
  const [viewingProcess, setViewingProcess] = useState<ConversationalProcess | null>(null);
  const [editingProcess, setEditingProcess] = useState<ConversationalProcess | null>(null);
  const [currentTab, setCurrentTab] = useState('processes');

  // New Process Form
  const [newProcess, setNewProcess] = useState({
    name: '',
    description: '',
    triggers: '',
    keywords: ''
  });

  // New Connector Form
  const [newConnector, setNewConnector] = useState({
    name: '',
    description: '',
    type: 'http',
    base_url: '',
    auth_type: 'none'
  });

  // Activity and Slot editing states (for future use)
  const [editingActivity, setEditingActivity] = useState<any>(null);
  const [editingSlot, setEditingSlot] = useState<any>(null);
  const [showActivityModal, setShowActivityModal] = useState(false);
  const [showSlotModal, setShowSlotModal] = useState(false);

  useEffect(() => {
    loadProcesses();
    loadConnectors();
  }, []);

  const loadProcesses = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/agent-studio/processes');
      const data = await response.json();
      setProcesses(data.processes || []);
    } catch (error) {
      console.error('Failed to load processes:', error);
    }
  };

  const loadConnectors = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/agent-studio/connectors');
      const data = await response.json();
      setConnectors(data.connectors || []);
      setLoading(false);
    } catch (error) {
      console.error('Failed to load connectors:', error);
      setLoading(false);
    }
  };

  const createProcess = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/agent-studio/processes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newProcess.name,
          description: newProcess.description,
          triggers: newProcess.triggers.split(',').map(t => t.trim()),
          keywords: newProcess.keywords.split(',').map(k => k.trim())
        })
      });
      
      if (response.ok) {
        setNewProcess({ name: '', description: '', triggers: '', keywords: '' });
        loadProcesses();
      }
    } catch (error) {
      console.error('Failed to create process:', error);
    }
  };

  const createConnector = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/agent-studio/connectors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newConnector)
      });

      if (response.ok) {
        setNewConnector({ name: '', description: '', type: 'http', base_url: '', auth_type: 'none' });
        loadConnectors();
      }
    } catch (error) {
      console.error('Failed to create connector:', error);
    }
  };

  const updateProcess = async (processData: any) => {
    if (!editingProcess) return;

    try {
      const response = await fetch(`http://localhost:8000/api/agent-studio/processes/${editingProcess.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(processData)
      });

      if (response.ok) {
        setEditingProcess(null);
        loadProcesses();
      }
    } catch (error) {
      console.error('Failed to update process:', error);
    }
  };

  const testProcess = async (processId: string) => {
    if (!testInput.trim()) return;
    
    setTesting(true);
    try {
      const response = await fetch('http://localhost:8000/api/agent-studio/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          process_id: processId,
          test_input: testInput
        })
      });
      
      const result = await response.json();
      if (response.ok) {
        loadTestResults(processId);
        setTestInput('');
      }
    } catch (error) {
      console.error('Failed to test process:', error);
    } finally {
      setTesting(false);
    }
  };

  const loadTestResults = async (processId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/agent-studio/processes/${processId}/test-results`);
      const data = await response.json();
      setTestResults(data.test_results || []);
    } catch (error) {
      console.error('Failed to load test results:', error);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'published': return 'bg-green-100 text-green-800';
      case 'testing': return 'bg-yellow-100 text-yellow-800';
      case 'draft': return 'bg-gray-100 text-gray-800';
      case 'archived': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const handleViewProcess = (process: ConversationalProcess) => {
    setViewingProcess(process);
  };

  const handleEditProcess = (process: ConversationalProcess) => {
    setEditingProcess(process);
  };

  const handleTestProcess = (process: ConversationalProcess) => {
    setSelectedProcess(process);
    loadTestResults(process.id);
    setCurrentTab('testing');
  };

  const parseTriggers = (triggers: any): string[] => {
    if (Array.isArray(triggers)) return triggers;
    if (typeof triggers === 'string') {
      try {
        return JSON.parse(triggers);
      } catch {
        return [triggers];
      }
    }
    return [];
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-lg">Loading Agent Studio...</div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Agent Studio</h1>
          <p className="text-gray-600">Build and manage conversational processes</p>
        </div>
        <div className="flex gap-2">
          <Dialog>
            <DialogTrigger asChild>
              <Button>
                <Plus className="w-4 h-4 mr-2" />
                New Process
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create New Process</DialogTitle>
              </DialogHeader>
              <div className="space-y-4">
                <div>
                  <Label htmlFor="name">Name</Label>
                  <Input
                    id="name"
                    value={newProcess.name}
                    onChange={(e) => setNewProcess({...newProcess, name: e.target.value})}
                    placeholder="Process name"
                  />
                </div>
                <div>
                  <Label htmlFor="description">Description</Label>
                  <Textarea
                    id="description"
                    value={newProcess.description}
                    onChange={(e) => setNewProcess({...newProcess, description: e.target.value})}
                    placeholder="Process description"
                  />
                </div>
                <div>
                  <Label htmlFor="triggers">Triggers (comma-separated)</Label>
                  <Input
                    id="triggers"
                    value={newProcess.triggers}
                    onChange={(e) => setNewProcess({...newProcess, triggers: e.target.value})}
                    placeholder="reset password, forgot password"
                  />
                </div>
                <div>
                  <Label htmlFor="keywords">Keywords (comma-separated)</Label>
                  <Input
                    id="keywords"
                    value={newProcess.keywords}
                    onChange={(e) => setNewProcess({...newProcess, keywords: e.target.value})}
                    placeholder="password, reset, help"
                  />
                </div>
                <Button onClick={createProcess} className="w-full">
                  Create Process
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      <Tabs value={currentTab} onValueChange={setCurrentTab} className="space-y-4">
        <TabsList>
          <TabsTrigger value="processes">
            <Code className="w-4 h-4 mr-2" />
            Processes ({processes.length})
          </TabsTrigger>
          <TabsTrigger value="connectors">
            <Database className="w-4 h-4 mr-2" />
            Connectors ({connectors.length})
          </TabsTrigger>
          <TabsTrigger value="testing">
            <TestTube className="w-4 h-4 mr-2" />
            Testing
          </TabsTrigger>
        </TabsList>

        <TabsContent value="processes" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {processes.map((process) => (
              <Card key={process.id} className="hover:shadow-md transition-shadow">
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between">
                    <div>
                      <CardTitle className="text-lg">{process.name}</CardTitle>
                      <p className="text-sm text-gray-600 mt-1">{process.description}</p>
                    </div>
                    <Badge className={getStatusColor(process.status)}>
                      {process.status}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span>Activities: {process.activities_count}</span>
                    <span>Slots: {process.slots_count}</span>
                  </div>
                  
                  <div className="flex flex-wrap gap-1">
                    {parseTriggers(process.triggers).slice(0, 2).map((trigger, idx) => (
                      <Badge key={idx} variant="outline" className="text-xs">
                        {trigger}
                      </Badge>
                    ))}
                    {parseTriggers(process.triggers).length > 2 && (
                      <Badge variant="outline" className="text-xs">
                        +{parseTriggers(process.triggers).length - 2} more
                      </Badge>
                    )}
                  </div>

                  <div className="flex gap-2 pt-2">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleViewProcess(process)}
                    >
                      <Eye className="w-3 h-3 mr-1" />
                      View
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleEditProcess(process)}
                    >
                      <Edit className="w-3 h-3 mr-1" />
                      Edit
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleTestProcess(process)}
                    >
                      <Play className="w-3 h-3 mr-1" />
                      Test
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="connectors" className="space-y-4">
          <div className="flex justify-end">
            <Dialog>
              <DialogTrigger asChild>
                <Button variant="outline">
                  <Plus className="w-4 h-4 mr-2" />
                  New Connector
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Create New Connector</DialogTitle>
                </DialogHeader>
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="conn-name">Name</Label>
                    <Input
                      id="conn-name"
                      value={newConnector.name}
                      onChange={(e) => setNewConnector({...newConnector, name: e.target.value})}
                      placeholder="Connector name"
                    />
                  </div>
                  <div>
                    <Label htmlFor="conn-description">Description</Label>
                    <Textarea
                      id="conn-description"
                      value={newConnector.description}
                      onChange={(e) => setNewConnector({...newConnector, description: e.target.value})}
                      placeholder="Connector description"
                    />
                  </div>
                  <div>
                    <Label htmlFor="conn-type">Type</Label>
                    <Select value={newConnector.type} onValueChange={(value) => setNewConnector({...newConnector, type: value})}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="http">HTTP</SelectItem>
                        <SelectItem value="database">Database</SelectItem>
                        <SelectItem value="api">API</SelectItem>
                        <SelectItem value="system">System</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label htmlFor="conn-url">Base URL</Label>
                    <Input
                      id="conn-url"
                      value={newConnector.base_url}
                      onChange={(e) => setNewConnector({...newConnector, base_url: e.target.value})}
                      placeholder="https://api.example.com"
                    />
                  </div>
                  <Button onClick={createConnector} className="w-full">
                    Create Connector
                  </Button>
                </div>
              </DialogContent>
            </Dialog>
          </div>

          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {connectors.map((connector) => (
              <Card key={connector.id} className="hover:shadow-md transition-shadow">
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between">
                    <div>
                      <CardTitle className="text-lg">{connector.name}</CardTitle>
                      <p className="text-sm text-gray-600 mt-1">{connector.description}</p>
                    </div>
                    <Badge className={getStatusColor(connector.status)}>
                      {connector.status}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span>Type: {connector.type}</span>
                    <span>Actions: {connector.actions_count}</span>
                  </div>
                  
                  <div className="text-xs text-gray-500 truncate">
                    {connector.base_url}
                  </div>

                  <div className="flex gap-2 pt-2">
                    <Button size="sm" variant="outline">
                      <Settings className="w-3 h-3 mr-1" />
                      Configure
                    </Button>
                    <Button size="sm" variant="outline">
                      <TestTube className="w-3 h-3 mr-1" />
                      Test
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="testing" className="space-y-4">
          {selectedProcess ? (
            <div className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle>Testing: {selectedProcess.name}</CardTitle>
                  <p className="text-sm text-gray-600">{selectedProcess.description}</p>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex gap-2">
                    <Input
                      value={testInput}
                      onChange={(e) => setTestInput(e.target.value)}
                      placeholder="Enter test input..."
                      className="flex-1"
                    />
                    <Button 
                      onClick={() => testProcess(selectedProcess.id)}
                      disabled={testing || !testInput.trim()}
                    >
                      {testing ? 'Testing...' : 'Test'}
                    </Button>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Test Results ({testResults.length})</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {testResults.map((result, idx) => (
                      <div key={idx} className="border rounded-lg p-3">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium">"{result.test_input}"</span>
                          <div className="flex items-center gap-2">
                            <Badge className={result.success ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}>
                              {result.success ? 'Success' : 'Failed'}
                            </Badge>
                            <span className="text-xs text-gray-500">
                              {result.execution_time.toFixed(3)}s
                            </span>
                          </div>
                        </div>
                        <p className="text-sm text-gray-600">{result.response}</p>
                        {result.steps_executed.length > 0 && (
                          <div className="mt-2 text-xs text-gray-500">
                            {result.steps_executed.length} steps executed
                          </div>
                        )}
                      </div>
                    ))}
                    {testResults.length === 0 && (
                      <p className="text-gray-500 text-center py-4">No test results yet</p>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
          ) : (
            <Card>
              <CardContent className="text-center py-8">
                <TestTube className="w-12 h-12 mx-auto text-gray-400 mb-4" />
                <p className="text-gray-600">Select a process to start testing</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>

      {/* View Process Modal */}
      <Dialog open={!!viewingProcess} onOpenChange={() => setViewingProcess(null)}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Process Details: {viewingProcess?.name}</DialogTitle>
          </DialogHeader>
          {viewingProcess && (
            <div className="space-y-4">
              <div>
                <Label>Description</Label>
                <p className="text-sm text-gray-600">{viewingProcess.description}</p>
              </div>

              <div>
                <Label>Status</Label>
                <Badge className={getStatusColor(viewingProcess.status)}>
                  {viewingProcess.status}
                </Badge>
              </div>

              <div>
                <Label>Triggers</Label>
                <div className="flex flex-wrap gap-1 mt-1">
                  {parseTriggers(viewingProcess.triggers).map((trigger, idx) => (
                    <Badge key={idx} variant="outline" className="text-xs">
                      {trigger}
                    </Badge>
                  ))}
                </div>
              </div>

              <div>
                <Label>Keywords</Label>
                <div className="flex flex-wrap gap-1 mt-1">
                  {parseTriggers(viewingProcess.keywords).map((keyword, idx) => (
                    <Badge key={idx} variant="outline" className="text-xs">
                      {keyword}
                    </Badge>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>Activities</Label>
                  <p className="text-sm text-gray-600">{viewingProcess.activities_count}</p>
                </div>
                <div>
                  <Label>Slots</Label>
                  <p className="text-sm text-gray-600">{viewingProcess.slots_count}</p>
                </div>
              </div>

              <div>
                <Label>Created</Label>
                <p className="text-sm text-gray-600">
                  {new Date(viewingProcess.created_at).toLocaleDateString()} by {viewingProcess.created_by}
                </p>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Advanced Process Editor */}
      <ProcessEditor
        process={editingProcess}
        isOpen={!!editingProcess}
        onClose={() => setEditingProcess(null)}
        onSave={updateProcess}
      />
    </div>
  );
};

export default AgentStudio;
