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
  Eye,
  Activity,
  Zap,
  GitBranch
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

interface SlotDefinition {
  id?: string;
  name: string;
  data_type: string;
  description: string;
  inference_policy: string;
  validation_policy?: string;
  validation_description?: string;
  resolver_strategy: {
    type: string;
    config: any;
  };
}

interface ActivityDefinition {
  id?: string;
  type: 'action' | 'content' | 'decision';
  name: string;
  config: any;
  input_mapping?: Record<string, string>;
  output_mapping?: Record<string, string>;
  confirmation_policy?: string;
  required_slots?: string[];
}

interface DeploymentConfig {
  environment: string;
  enabled: boolean;
  user_groups: string[];
  rollback_plan?: string;
}

const AgentStudio: React.FC = () => {
  const [processes, setProcesses] = useState<ConversationalProcess[]>([]);
  const [connectors, setConnectors] = useState<Connector[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedProcess, setSelectedProcess] = useState<ConversationalProcess | null>(null);
  const [testResults, setTestResults] = useState<TestResult[]>([]);
  const [testInput, setTestInput] = useState('');
  const [testing, setTesting] = useState(false);

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

  // Slot Configuration
  const [slots, setSlots] = useState<SlotDefinition[]>([]);
  const [newSlot, setNewSlot] = useState<SlotDefinition>({
    name: '',
    data_type: 'string',
    description: '',
    inference_policy: 'infer',
    resolver_strategy: {
      type: 'static',
      config: { options: [] }
    }
  });

  // Activity Configuration
  const [activities, setActivities] = useState<ActivityDefinition[]>([]);
  const [newActivity, setNewActivity] = useState<ActivityDefinition>({
    type: 'action',
    name: '',
    config: {}
  });

  // Deployment Configuration
  const [deployments, setDeployments] = useState<DeploymentConfig[]>([]);
  const [newDeployment, setNewDeployment] = useState<DeploymentConfig>({
    environment: 'staging',
    enabled: true,
    user_groups: []
  });

  // Dialog states
  const [showSlotDialog, setShowSlotDialog] = useState(false);
  const [showActivityDialog, setShowActivityDialog] = useState(false);
  const [showDeployDialog, setShowDeployDialog] = useState(false);

  useEffect(() => {
    console.log('🔄 Loading Agent Studio data...');
    console.log('📊 Current state:', {
      processes: processes.length,
      connectors: connectors.length,
      slots: slots.length,
      activities: activities.length,
      deployments: deployments.length
    });
    loadProcesses();
    loadConnectors();
    loadSlots();
    loadActivities();
    loadDeployments();
  }, []);

  // Debug effect to track state changes
  useEffect(() => {
    console.log('🔍 State updated:', {
      processes: processes.length,
      connectors: connectors.length,
      slots: slots.length,
      activities: activities.length,
      deployments: deployments.length
    });
  }, [processes, connectors, slots, activities, deployments]);

  const loadProcesses = async () => {
    try {
      const response = await fetch('/api/agent-studio/processes');
      const data = await response.json();
      setProcesses(data.processes || []);
    } catch (error) {
      console.error('Failed to load processes:', error);
    }
  };

  const loadConnectors = async () => {
    try {
      const response = await fetch('/api/agent-studio/connectors');
      const data = await response.json();
      setConnectors(data.connectors || []);
      setLoading(false);
    } catch (error) {
      console.error('Failed to load connectors:', error);
      setLoading(false);
    }
  };

  const loadSlots = async () => {
    try {
      console.log('📊 Loading slots...');
      const response = await fetch('/api/agent-studio/slots');
      const data = await response.json();
      console.log('📊 Slots loaded:', data.slots?.length || 0);
      setSlots(data.slots || []);
    } catch (error) {
      console.error('Failed to load slots:', error);
    }
  };

  const loadActivities = async () => {
    try {
      const response = await fetch('/api/agent-studio/activities');
      const data = await response.json();
      setActivities(data.activities || []);
    } catch (error) {
      console.error('Failed to load activities:', error);
    }
  };

  const loadDeployments = async () => {
    try {
      const response = await fetch('/api/agent-studio/deployments');
      const data = await response.json();
      setDeployments(data.deployments || []);
    } catch (error) {
      console.error('Failed to load deployments:', error);
    }
  };

  const createProcess = async () => {
    try {
      const response = await fetch('/api/agent-studio/processes', {
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
      const response = await fetch('/api/agent-studio/connectors', {
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

  const testProcess = async (processId: string) => {
    if (!testInput.trim()) return;
    
    setTesting(true);
    try {
      const response = await fetch('/api/agent-studio/test', {
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
      const response = await fetch(`/api/agent-studio/processes/${processId}/test-results`);
      const data = await response.json();
      setTestResults(data.test_results || []);
    } catch (error) {
      console.error('Failed to load test results:', error);
    }
  };

  const createSlot = async () => {
    try {
      const response = await fetch('/api/agent-studio/slots', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newSlot)
      });

      if (response.ok) {
        const slot = await response.json();
        setSlots([...slots, slot]);
        setNewSlot({
          name: '',
          data_type: 'string',
          description: '',
          inference_policy: 'infer',
          resolver_strategy: { type: 'static', config: { options: [] } }
        });
        setShowSlotDialog(false);
      }
    } catch (error) {
      console.error('Failed to create slot:', error);
    }
  };

  const createActivity = async () => {
    try {
      const response = await fetch('/api/agent-studio/activities', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newActivity)
      });

      if (response.ok) {
        const activity = await response.json();
        setActivities([...activities, activity]);
        setNewActivity({
          type: 'action',
          name: '',
          config: {}
        });
        setShowActivityDialog(false);
      }
    } catch (error) {
      console.error('Failed to create activity:', error);
    }
  };

  const deployProcess = async () => {
    try {
      const response = await fetch('/api/agent-studio/deploy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          process_id: selectedProcess?.id,
          ...newDeployment
        })
      });

      if (response.ok) {
        const deployment = await response.json();
        setDeployments([...deployments, deployment]);
        setNewDeployment({
          environment: 'staging',
          enabled: true,
          user_groups: []
        });
        setShowDeployDialog(false);
      }
    } catch (error) {
      console.error('Failed to deploy process:', error);
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

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-lg">Loading Agent Studio...</div>
      </div>
    );
  }

  // Debug: Log render state
  console.log('🎨 Rendering Agent Studio with:', {
    processes: processes.length,
    slots: slots.length,
    activities: activities.length,
    connectors: connectors.length,
    deployments: deployments.length
  });

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

      <Tabs defaultValue="processes" className="space-y-4">
        <TabsList className="flex w-full overflow-x-auto space-x-1">
          <TabsTrigger value="processes" className="flex-shrink-0">
            <Code className="w-4 h-4 mr-2" />
            Processes ({processes.length})
          </TabsTrigger>
          <TabsTrigger value="slots" className="flex-shrink-0">
            <Zap className="w-4 h-4 mr-2" />
            Slots ({slots.length})
          </TabsTrigger>
          <TabsTrigger value="activities" className="flex-shrink-0">
            <Activity className="w-4 h-4 mr-2" />
            Activities ({activities.length})
          </TabsTrigger>
          <TabsTrigger value="connectors" className="flex-shrink-0">
            <Database className="w-4 h-4 mr-2" />
            Connectors ({connectors.length})
          </TabsTrigger>
          <TabsTrigger value="testing" className="flex-shrink-0">
            <TestTube className="w-4 h-4 mr-2" />
            Testing
          </TabsTrigger>
          <TabsTrigger value="deployment" className="flex-shrink-0">
            <Rocket className="w-4 h-4 mr-2" />
            Deploy
          </TabsTrigger>
        </TabsList>

        <TabsContent value="slots" className="space-y-4">
          <div className="flex justify-between items-center">
            <h2 className="text-xl font-semibold">Slot Definitions</h2>
            <Dialog open={showSlotDialog} onOpenChange={setShowSlotDialog}>
              <DialogTrigger asChild>
                <Button>
                  <Plus className="w-4 h-4 mr-2" />
                  New Slot
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-2xl">
                <DialogHeader>
                  <DialogTitle>Create New Slot</DialogTitle>
                </DialogHeader>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="slot-name">Slot Name</Label>
                      <Input
                        id="slot-name"
                        value={newSlot.name}
                        onChange={(e) => setNewSlot({...newSlot, name: e.target.value})}
                        placeholder="e.g., pto_type"
                      />
                    </div>
                    <div>
                      <Label htmlFor="slot-type">Data Type</Label>
                      <Select value={newSlot.data_type} onValueChange={(value) => setNewSlot({...newSlot, data_type: value})}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="string">String</SelectItem>
                          <SelectItem value="number">Number</SelectItem>
                          <SelectItem value="boolean">Boolean</SelectItem>
                          <SelectItem value="object">Custom Object</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div>
                    <Label htmlFor="slot-description">Description</Label>
                    <Textarea
                      id="slot-description"
                      value={newSlot.description}
                      onChange={(e) => setNewSlot({...newSlot, description: e.target.value})}
                      placeholder="Describe what this slot represents"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="inference-policy">Inference Policy</Label>
                      <Select value={newSlot.inference_policy} onValueChange={(value) => setNewSlot({...newSlot, inference_policy: value})}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="infer">Infer if available</SelectItem>
                          <SelectItem value="always_ask">Always ask explicitly</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label htmlFor="resolver-type">Resolver Type</Label>
                      <Select value={newSlot.resolver_strategy.type} onValueChange={(value) => setNewSlot({...newSlot, resolver_strategy: {...newSlot.resolver_strategy, type: value}})}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="static">Static Options</SelectItem>
                          <SelectItem value="api">API Call</SelectItem>
                          <SelectItem value="vector_search">Vector Search</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div>
                    <Label htmlFor="validation-policy">Validation Policy (Optional)</Label>
                    <Input
                      id="validation-policy"
                      value={newSlot.validation_policy || ''}
                      onChange={(e) => setNewSlot({...newSlot, validation_policy: e.target.value})}
                      placeholder="e.g., value IN ['vacation', 'sick', 'personal']"
                    />
                  </div>

                  <Button onClick={createSlot} className="w-full">
                    Create Slot
                  </Button>
                </div>
              </DialogContent>
            </Dialog>
          </div>

          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {slots.map((slot) => (
              <Card key={slot.id || slot.name} className="hover:shadow-md transition-shadow">
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between">
                    <div>
                      <CardTitle className="text-lg">{slot.name}</CardTitle>
                      <p className="text-sm text-gray-600 mt-1">{slot.description}</p>
                    </div>
                    <Badge variant="outline">{slot.data_type}</Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span>Policy: {slot.inference_policy}</span>
                    <span>Resolver: {slot.resolver_strategy.type}</span>
                  </div>

                  {slot.validation_policy && (
                    <div className="text-xs text-gray-500 bg-gray-50 p-2 rounded">
                      Validation: {slot.validation_policy}
                    </div>
                  )}

                  <div className="flex gap-2 pt-2">
                    <Button size="sm" variant="outline">
                      <Edit className="w-3 h-3 mr-1" />
                      Edit
                    </Button>
                    <Button size="sm" variant="outline">
                      <Trash2 className="w-3 h-3 mr-1" />
                      Delete
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
            {slots.length === 0 && (
              <div className="col-span-full text-center py-8 text-gray-500">
                <Zap className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                <p>No slots defined yet. Create your first slot to get started.</p>
              </div>
            )}
          </div>
        </TabsContent>

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
                    {Array.isArray(process.triggers) ? process.triggers.slice(0, 2).map((trigger, idx) => (
                      <Badge key={idx} variant="outline" className="text-xs">
                        {trigger}
                      </Badge>
                    )) : null}
                    {Array.isArray(process.triggers) && process.triggers.length > 2 && (
                      <Badge variant="outline" className="text-xs">
                        +{process.triggers.length - 2} more
                      </Badge>
                    )}
                  </div>

                  <div className="flex gap-2 pt-2">
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => {
                        setSelectedProcess(process);
                        loadTestResults(process.id);
                      }}
                    >
                      <Eye className="w-3 h-3 mr-1" />
                      View
                    </Button>
                    <Button size="sm" variant="outline">
                      <Edit className="w-3 h-3 mr-1" />
                      Edit
                    </Button>
                    <Button size="sm" variant="outline">
                      <Play className="w-3 h-3 mr-1" />
                      Test
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="activities" className="space-y-4">
          <div className="flex justify-between items-center">
            <h2 className="text-xl font-semibold">Activity Definitions</h2>
            <Dialog open={showActivityDialog} onOpenChange={setShowActivityDialog}>
              <DialogTrigger asChild>
                <Button>
                  <Plus className="w-4 h-4 mr-2" />
                  New Activity
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
                <DialogHeader>
                  <DialogTitle>Create New Activity - Moveworks Style</DialogTitle>
                  <p className="text-sm text-gray-600">Configure activities that represent business process steps</p>
                </DialogHeader>
                <div className="space-y-6">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="activity-name">Activity Name</Label>
                      <Input
                        id="activity-name"
                        value={newActivity.name}
                        onChange={(e) => setNewActivity({...newActivity, name: e.target.value})}
                        placeholder="e.g., get_pto_balance_activity"
                      />
                    </div>
                    <div>
                      <Label htmlFor="activity-type">Activity Type</Label>
                      <Select value={newActivity.type} onValueChange={(value: 'action' | 'content' | 'decision') => setNewActivity({...newActivity, type: value})}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="action">Action Activity</SelectItem>
                          <SelectItem value="content">Content Activity</SelectItem>
                          <SelectItem value="decision">Decision Activity</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  {newActivity.type === 'action' && (
                    <div className="space-y-6 border rounded-lg p-6 bg-blue-50">
                      <div className="flex items-center gap-2">
                        <Settings className="w-5 h-5 text-blue-600" />
                        <h3 className="font-semibold text-blue-900">Action Activity Configuration</h3>
                      </div>
                      <p className="text-sm text-blue-700">Action Activities call HTTP, Script, Built-in, or Compound Actions to fetch or update data in your business systems.</p>

                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <Label htmlFor="action-name">Select an Action *</Label>
                          <Select onValueChange={(value) => setNewActivity({...newActivity, config: {...newActivity.config, action_name: value}})}>
                            <SelectTrigger>
                              <SelectValue placeholder="Choose action to execute" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="get_pto_balance_action">Get PTO Balance (HTTP)</SelectItem>
                              <SelectItem value="update_feature_request_action">Update Feature Request (HTTP)</SelectItem>
                              <SelectItem value="submit_purchase_request_action">Submit Purchase Request (HTTP)</SelectItem>
                              <SelectItem value="generate_text_action">Generate Text (Built-in AI)</SelectItem>
                              <SelectItem value="create_ticket_action">Create ServiceNow Ticket (HTTP)</SelectItem>
                              <SelectItem value="lookup_user_action">Lookup User Info (Script)</SelectItem>
                            </SelectContent>
                          </Select>
                          <p className="text-xs text-gray-500 mt-1">Dropdown of all available actions (HTTP, Script, Built-in, Compound)</p>
                        </div>

                        <div>
                          <Label htmlFor="output-key">Output Key *</Label>
                          <Input
                            id="output-key"
                            value={newActivity.config?.output_key || ''}
                            onChange={(e) => setNewActivity({...newActivity, config: {...newActivity.config, output_key: e.target.value}})}
                            placeholder="e.g., pto_balance_result"
                          />
                          <p className="text-xs text-gray-500 mt-1">Keyword that represents the output of this Activity</p>
                        </div>
                      </div>

                      <div>
                        <Label htmlFor="required-slots">Required Slots</Label>
                        <Input
                          id="required-slots"
                          value={newActivity.required_slots?.join(', ') || ''}
                          onChange={(e) => setNewActivity({...newActivity, required_slots: e.target.value.split(',').map(s => s.trim()).filter(Boolean)})}
                          placeholder="e.g., pto_type, user_email"
                        />
                        <p className="text-xs text-gray-500 mt-1">Defines which slots are required to run this Activity. Tells AI when to ask user for values.</p>
                      </div>

                      <div>
                        <Label htmlFor="input-mapping">Input Mapper (JSON Bender)</Label>
                        <Textarea
                          id="input-mapping"
                          value={JSON.stringify(newActivity.input_mapping || {}, null, 2)}
                          onChange={(e) => {
                            try {
                              setNewActivity({...newActivity, input_mapping: JSON.parse(e.target.value)});
                            } catch {}
                          }}
                          placeholder='{\n  "pto_type": "data.pto_type.value",\n  "user_email": "meta_info.user.email_addr",\n  "employee_id": "meta_info.user.employee_id"\n}'
                          rows={5}
                        />
                        <p className="text-xs text-gray-500 mt-1">Maps data (slots or outputs from previous Activities) to the inputs for your action</p>
                      </div>

                      <div>
                        <Label htmlFor="output-mapping">Output Mapper</Label>
                        <Textarea
                          id="output-mapping"
                          value={JSON.stringify(newActivity.output_mapping || {}, null, 2)}
                          onChange={(e) => {
                            try {
                              setNewActivity({...newActivity, output_mapping: JSON.parse(e.target.value)});
                            } catch {}
                          }}
                          placeholder='{\n  "available_days": "response.data.available_days",\n  "total_days": "response.data.total_days",\n  "balance_type": "response.data.balance_type"\n}'
                          rows={4}
                        />
                        <p className="text-xs text-gray-500 mt-1">Represents the output object. By default captures all data. Use "dot walk" to limit data returned.</p>
                      </div>
                    </div>
                  )}

                  {newActivity.type === 'content' && (
                    <div className="space-y-6 border rounded-lg p-6 bg-green-50">
                      <div className="flex items-center gap-2">
                        <Eye className="w-5 h-5 text-green-600" />
                        <h3 className="font-semibold text-green-900">Content Activity Configuration</h3>
                      </div>
                      <p className="text-sm text-green-700">Content activities share articles, forms, or markdown text with users. They do not require inputs to run and do not produce data.</p>

                      <div>
                        <Label htmlFor="content-type">Content Type</Label>
                        <Select value={newActivity.config?.content_type || 'text'} onValueChange={(value) => setNewActivity({...newActivity, config: {...newActivity.config, content_type: value}})}>
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="text">Text/Markdown</SelectItem>
                            <SelectItem value="article">Knowledge Article</SelectItem>
                            <SelectItem value="form">Form</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div>
                        <Label htmlFor="content-text">Content Text</Label>
                        <Textarea
                          id="content-text"
                          value={newActivity.config?.content_text || ''}
                          onChange={(e) => setNewActivity({...newActivity, config: {...newActivity.config, content_text: e.target.value}})}
                          placeholder="Your {{pto_type}} balance is {{pto_balance_result.available_days}} days available out of {{pto_balance_result.total_days}} total days."
                          rows={6}
                        />
                        <div className="mt-2 p-3 bg-yellow-50 border border-yellow-200 rounded">
                          <h4 className="font-medium text-yellow-800 text-sm mb-2">Dynamic Data Reference (Mustache Syntax)</h4>
                          <ul className="text-xs text-yellow-700 space-y-1">
                            <li>• Use <code>{{`{key}`}}</code> to reference data from previous activities or slots</li>
                            <li>• Example: <code>{{`{ticket_number}`}}</code> references the ticket_number key</li>
                            <li>• For arrays: <code>{{`{ticket_list.0}`}}</code> accesses first element</li>
                            <li>• No "data." prefix needed - points directly to entire data bank</li>
                            <li>• Only strings supported, not arrays or objects</li>
                          </ul>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <Label htmlFor="content-title">Content Title (Optional)</Label>
                          <Input
                            id="content-title"
                            value={newActivity.config?.content_title || ''}
                            onChange={(e) => setNewActivity({...newActivity, config: {...newActivity.config, content_title: e.target.value}})}
                            placeholder="PTO Balance Information"
                          />
                        </div>
                        <div>
                          <Label htmlFor="content-format">Format</Label>
                          <Select value={newActivity.config?.content_format || 'markdown'} onValueChange={(value) => setNewActivity({...newActivity, config: {...newActivity.config, content_format: value}})}>
                            <SelectTrigger>
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="markdown">Markdown</SelectItem>
                              <SelectItem value="plain">Plain Text</SelectItem>
                              <SelectItem value="html">HTML</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      </div>
                    </div>
                  )}

                  {newActivity.type === 'decision' && (
                    <div className="space-y-6 border rounded-lg p-6 bg-purple-50">
                      <div className="flex items-center gap-2">
                        <GitBranch className="w-5 h-5 text-purple-600" />
                        <h3 className="font-semibold text-purple-900">Decision Activity Configuration</h3>
                      </div>
                      <p className="text-sm text-purple-700">Decision activities create conditional branching in your process flow based on data values or business logic.</p>

                      <div>
                        <Label htmlFor="decision-logic">Decision Logic Type</Label>
                        <Select value={newActivity.config?.decision_logic || 'conditional'} onValueChange={(value) => setNewActivity({...newActivity, config: {...newActivity.config, decision_logic: value}})}>
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="conditional">Conditional Branching</SelectItem>
                            <SelectItem value="switch">Switch Statement</SelectItem>
                            <SelectItem value="ai_classification">AI Classification</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div>
                        <Label htmlFor="decision-cases">Decision Cases (JSON)</Label>
                        <Textarea
                          id="decision-cases"
                          value={JSON.stringify(newActivity.config?.decision_cases || [], null, 2)}
                          onChange={(e) => {
                            try {
                              setNewActivity({...newActivity, config: {...newActivity.config, decision_cases: JSON.parse(e.target.value)}});
                            } catch {}
                          }}
                          placeholder='[\n  {\n    "condition": "data.classification == \'opex\'",\n    "next_activity": 4,\n    "description": "OPEX requests go to finance"\n  },\n  {\n    "condition": "data.amount > 1000",\n    "next_activity": 5,\n    "description": "High value requests need approval"\n  }\n]'
                          rows={8}
                        />
                        <div className="mt-2 p-3 bg-blue-50 border border-blue-200 rounded">
                          <h4 className="font-medium text-blue-800 text-sm mb-2">Decision Case Structure</h4>
                          <ul className="text-xs text-blue-700 space-y-1">
                            <li>• <strong>condition</strong>: DSL expression to evaluate (e.g., "data.amount > 1000")</li>
                            <li>• <strong>next_activity</strong>: Activity ID to route to if condition is true</li>
                            <li>• <strong>description</strong>: Human-readable explanation of the condition</li>
                            <li>• Cases are evaluated in order - first match wins</li>
                          </ul>
                        </div>
                      </div>

                      <div>
                        <Label htmlFor="default-activity">Default Activity (Fallback)</Label>
                        <Input
                          id="default-activity"
                          value={newActivity.config?.default_activity || ''}
                          onChange={(e) => setNewActivity({...newActivity, config: {...newActivity.config, default_activity: e.target.value}})}
                          placeholder="Activity ID for when no conditions match"
                        />
                        <p className="text-xs text-gray-500 mt-1">Activity to route to if none of the decision cases match</p>
                      </div>

                      <div>
                        <Label htmlFor="decision-variable">Decision Variable</Label>
                        <Input
                          id="decision-variable"
                          value={newActivity.config?.decision_variable || ''}
                          onChange={(e) => setNewActivity({...newActivity, config: {...newActivity.config, decision_variable: e.target.value}})}
                          placeholder="data.classification"
                        />
                        <p className="text-xs text-gray-500 mt-1">Primary variable to base decisions on (for switch-style logic)</p>
                      </div>
                    </div>
                  )}

                  <div className="space-y-4 border rounded-lg p-4 bg-orange-50">
                    <div className="flex items-center gap-2">
                      <Settings className="w-4 h-4 text-orange-600" />
                      <h4 className="font-medium text-orange-900">Confirmation Policy</h4>
                    </div>
                    <p className="text-sm text-orange-700">Control when users need to provide explicit consent before executing this activity.</p>

                    <div>
                      <Label htmlFor="confirmation-policy">Confirmation Required</Label>
                      <Select value={newActivity.confirmation_policy || 'none'} onValueChange={(value) => setNewActivity({...newActivity, confirmation_policy: value})}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="none">No confirmation required</SelectItem>
                          <SelectItem value="required">Require explicit user consent</SelectItem>
                          <SelectItem value="conditional">Conditional confirmation</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    {newActivity.confirmation_policy === 'required' && (
                      <div>
                        <Label htmlFor="confirmation-message">Confirmation Message</Label>
                        <Textarea
                          id="confirmation-message"
                          value={newActivity.config?.confirmation_message || ''}
                          onChange={(e) => setNewActivity({...newActivity, config: {...newActivity.config, confirmation_message: e.target.value}})}
                          placeholder="Are you sure you want to submit this PTO request for {{pto_type}} leave?"
                          rows={2}
                        />
                        <p className="text-xs text-gray-500 mt-1">Message shown to user before executing the activity</p>
                      </div>
                    )}

                    {newActivity.confirmation_policy === 'conditional' && (
                      <div>
                        <Label htmlFor="confirmation-condition">Confirmation Condition</Label>
                        <Input
                          id="confirmation-condition"
                          value={newActivity.config?.confirmation_condition || ''}
                          onChange={(e) => setNewActivity({...newActivity, config: {...newActivity.config, confirmation_condition: e.target.value}})}
                          placeholder="data.amount > 1000"
                        />
                        <p className="text-xs text-gray-500 mt-1">DSL condition for when confirmation is required</p>
                      </div>
                    )}
                  </div>

                  <Button onClick={createActivity} className="w-full">
                    Create Activity
                  </Button>
                </div>
              </DialogContent>
            </Dialog>
          </div>

          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {activities.map((activity) => (
              <Card key={activity.id || activity.name} className="hover:shadow-md transition-shadow border-l-4 border-l-blue-500">
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-2">
                      {activity.type === 'action' && <Settings className="w-5 h-5 text-blue-600" />}
                      {activity.type === 'content' && <Eye className="w-5 h-5 text-green-600" />}
                      {activity.type === 'decision' && <GitBranch className="w-5 h-5 text-purple-600" />}
                      <div>
                        <CardTitle className="text-lg">{activity.name}</CardTitle>
                        <p className="text-sm text-gray-600 mt-1 capitalize">{activity.type} Activity</p>
                      </div>
                    </div>
                    <Badge className={
                      activity.type === 'action' ? 'bg-blue-100 text-blue-800' :
                      activity.type === 'content' ? 'bg-green-100 text-green-800' :
                      'bg-purple-100 text-purple-800'
                    }>
                      {activity.type.toUpperCase()}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-3">
                  {activity.type === 'action' && (
                    <div className="space-y-2">
                      <div className="text-sm">
                        <p className="font-medium text-gray-700">Action:</p>
                        <p className="text-blue-600">{activity.config?.action_name || 'Not configured'}</p>
                      </div>
                      <div className="text-sm">
                        <p className="font-medium text-gray-700">Required Slots:</p>
                        <p className="text-gray-600">{activity.required_slots?.join(', ') || 'None'}</p>
                      </div>
                      <div className="text-sm">
                        <p className="font-medium text-gray-700">Output Key:</p>
                        <p className="text-gray-600">{activity.config?.output_key || 'Not set'}</p>
                      </div>
                    </div>
                  )}

                  {activity.type === 'content' && (
                    <div className="space-y-2">
                      <div className="text-sm">
                        <p className="font-medium text-gray-700">Content Type:</p>
                        <p className="text-green-600 capitalize">{activity.config?.content_type || 'text'}</p>
                      </div>
                      <div className="text-sm">
                        <p className="font-medium text-gray-700">Preview:</p>
                        <p className="text-gray-600 line-clamp-2 bg-gray-50 p-2 rounded text-xs">
                          {activity.config?.content_text || 'No content configured'}
                        </p>
                      </div>
                    </div>
                  )}

                  {activity.type === 'decision' && (
                    <div className="space-y-2">
                      <div className="text-sm">
                        <p className="font-medium text-gray-700">Logic Type:</p>
                        <p className="text-purple-600 capitalize">{activity.config?.decision_logic || 'conditional'}</p>
                      </div>
                      <div className="text-sm">
                        <p className="font-medium text-gray-700">Decision Cases:</p>
                        <p className="text-gray-600">{activity.config?.decision_cases?.length || 0} conditions</p>
                      </div>
                    </div>
                  )}

                  <div className="flex flex-wrap gap-1 pt-2">
                    {activity.confirmation_policy === 'required' && (
                      <Badge variant="secondary" className="text-xs bg-orange-100 text-orange-800">
                        Requires Confirmation
                      </Badge>
                    )}
                    {activity.confirmation_policy === 'conditional' && (
                      <Badge variant="secondary" className="text-xs bg-yellow-100 text-yellow-800">
                        Conditional Confirmation
                      </Badge>
                    )}
                    {activity.required_slots && activity.required_slots.length > 0 && (
                      <Badge variant="secondary" className="text-xs bg-blue-100 text-blue-800">
                        {activity.required_slots.length} Slots Required
                      </Badge>
                    )}
                  </div>

                  <div className="flex gap-2 pt-2">
                    <Button size="sm" variant="outline" className="flex-1">
                      <Edit className="w-3 h-3 mr-1" />
                      Edit
                    </Button>
                    <Button size="sm" variant="outline" className="flex-1">
                      <Play className="w-3 h-3 mr-1" />
                      Test
                    </Button>
                    <Button size="sm" variant="outline" className="text-red-600 hover:text-red-700">
                      <Trash2 className="w-3 h-3" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
            {activities.length === 0 && (
              <div className="col-span-full text-center py-8 text-gray-500">
                <Activity className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                <p>No activities defined yet. Create your first activity to get started.</p>
              </div>
            )}
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
                      valueHUMAN_IN_LOOP_TOOLS={newConnector.name}
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

        <TabsContent value="deployment" className="space-y-4">
          <div className="flex justify-between items-center">
            <h2 className="text-xl font-semibold">Process Deployment</h2>
            <Dialog open={showDeployDialog} onOpenChange={setShowDeployDialog}>
              <DialogTrigger asChild>
                <Button disabled={!selectedProcess}>
                  <Rocket className="w-4 h-4 mr-2" />
                  Deploy Process
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-2xl">
                <DialogHeader>
                  <DialogTitle>Deploy Process: {selectedProcess?.name}</DialogTitle>
                </DialogHeader>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="environment">Environment</Label>
                      <Select value={newDeployment.environment} onValueChange={(value) => setNewDeployment({...newDeployment, environment: value})}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="development">Development</SelectItem>
                          <SelectItem value="staging">Staging</SelectItem>
                          <SelectItem value="production">Production</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="flex items-center space-x-2 pt-6">
                      <input
                        type="checkbox"
                        id="enabled"
                        checked={newDeployment.enabled}
                        onChange={(e) => setNewDeployment({...newDeployment, enabled: e.target.checked})}
                        className="rounded"
                      />
                      <Label htmlFor="enabled">Enable immediately</Label>
                    </div>
                  </div>

                  <div>
                    <Label htmlFor="user-groups">User Groups (comma-separated)</Label>
                    <Input
                      id="user-groups"
                      value={newDeployment.user_groups.join(', ')}
                      onChange={(e) => setNewDeployment({...newDeployment, user_groups: e.target.value.split(',').map(g => g.trim()).filter(Boolean)})}
                      placeholder="e.g., hr_team, managers, all_employees"
                    />
                  </div>

                  <div>
                    <Label htmlFor="rollback-plan">Rollback Plan (Optional)</Label>
                    <Textarea
                      id="rollback-plan"
                      value={newDeployment.rollback_plan || ''}
                      onChange={(e) => setNewDeployment({...newDeployment, rollback_plan: e.target.value})}
                      placeholder="Describe the rollback procedure if issues occur"
                      rows={3}
                    />
                  </div>

                  <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                    <h4 className="font-medium text-yellow-800 mb-2">Deployment Checklist</h4>
                    <ul className="text-sm text-yellow-700 space-y-1">
                      <li>✓ Process has been tested successfully</li>
                      <li>✓ All required slots are defined</li>
                      <li>✓ All activities are configured</li>
                      <li>✓ Connectors are active and tested</li>
                      <li>✓ User permissions are properly set</li>
                    </ul>
                  </div>

                  <Button onClick={deployProcess} className="w-full">
                    Deploy to {newDeployment.environment}
                  </Button>
                </div>
              </DialogContent>
            </Dialog>
          </div>

          {selectedProcess ? (
            <div className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle>Deployment Status: {selectedProcess.name}</CardTitle>
                  <p className="text-sm text-gray-600">{selectedProcess.description}</p>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-4 md:grid-cols-3">
                    {deployments.map((deployment, idx) => (
                      <div key={idx} className="border rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium capitalize">{deployment.environment}</span>
                          <Badge className={deployment.enabled ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'}>
                            {deployment.enabled ? 'Active' : 'Inactive'}
                          </Badge>
                        </div>
                        <div className="text-sm text-gray-600">
                          <div>Groups: {deployment.user_groups.join(', ') || 'All users'}</div>
                        </div>
                        <div className="flex gap-2 mt-3">
                          <Button size="sm" variant="outline">
                            <Settings className="w-3 h-3 mr-1" />
                            Configure
                          </Button>
                          <Button size="sm" variant="outline">
                            <GitBranch className="w-3 h-3 mr-1" />
                            Rollback
                          </Button>
                        </div>
                      </div>
                    ))}
                    {deployments.length === 0 && (
                      <div className="col-span-full text-center py-8 text-gray-500">
                        <Rocket className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                        <p>No deployments yet. Deploy this process to get started.</p>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Analytics & Monitoring</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-4 md:grid-cols-4">
                    <div className="text-center p-4 bg-blue-50 rounded-lg">
                      <div className="text-2xl font-bold text-blue-600">1,247</div>
                      <div className="text-sm text-blue-600">Total Conversations</div>
                    </div>
                    <div className="text-center p-4 bg-green-50 rounded-lg">
                      <div className="text-2xl font-bold text-green-600">94.2%</div>
                      <div className="text-sm text-green-600">Success Rate</div>
                    </div>
                    <div className="text-center p-4 bg-yellow-50 rounded-lg">
                      <div className="text-2xl font-bold text-yellow-600">2.3s</div>
                      <div className="text-sm text-yellow-600">Avg Response Time</div>
                    </div>
                    <div className="text-center p-4 bg-purple-50 rounded-lg">
                      <div className="text-2xl font-bold text-purple-600">4.7/5</div>
                      <div className="text-sm text-purple-600">User Satisfaction</div>
                    </div>
                  </div>

                  <div className="mt-4 flex gap-2">
                    <Button size="sm" variant="outline">
                      <BarChart3 className="w-3 h-3 mr-1" />
                      View Detailed Analytics
                    </Button>
                    <Button size="sm" variant="outline">
                      <Eye className="w-3 h-3 mr-1" />
                      Monitor Live Usage
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>
          ) : (
            <Card>
              <CardContent className="text-center py-8">
                <Rocket className="w-12 h-12 mx-auto text-gray-400 mb-4" />
                <p className="text-gray-600">Select a process to view deployment options</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AgentStudio;
