'use client';

import React, { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './ui/dialog';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Textarea } from './ui/textarea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Badge } from './ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Plus, Trash2, Settings, Code, Database, Users, TestTube } from 'lucide-react';

interface ProcessEditorProps {
  process: any;
  isOpen: boolean;
  onClose: () => void;
  onSave: (processData: any) => void;
}

interface Slot {
  id: string;
  name: string;
  data_type: 'string' | 'number' | 'boolean' | 'custom_data_type';
  slot_description: string;
  slot_validation_policy: string;
  slot_validation_description: string;
  slot_inference_policy: 'infer_if_available' | 'always_ask_explicitly';
  resolver_strategy?: {
    method_name: string;
    method_type: 'static' | 'api' | 'inline';
    static_options?: Array<{
      display_value: string;
      raw_value: string;
    }>;
  };
}

interface Activity {
  id: string;
  name: string;
  type: 'action_activity' | 'response_activity' | 'condition_activity' | 'slot_collection_activity' | 'connector_activity';
  action_reference?: string; // e.g., 'mw.generate_text_action', 'firstname_lastname_submit_purchase_request'
  required_slots: string[]; // Array of slot names
  input_mapping: Record<string, string>; // DSL expressions like 'data.item_name', '$CONCAT(...)'
  output_mapping: string; // Dot notation like '.openai_chat_completions_response.choices[0].message.content'
  output_key: string; // Key for storing in data bank
  confirmation_policy: 'none' | 'require_consent' | 'custom';
  description: string;
}

export default function ProcessEditor({ process, isOpen, onClose, onSave }: ProcessEditorProps) {
  const [editData, setEditData] = useState({
    name: process?.name || '',
    description: process?.description || '',
    triggers: Array.isArray(process?.triggers) ? process.triggers.join(', ') : '',
    keywords: Array.isArray(process?.keywords) ? process.keywords.join(', ') : '',
    slots: process?.slots || [],
    activities: process?.activities || [],
    permissions: process?.permissions || { user_groups: [], roles: [] },
    required_connectors: process?.required_connectors || []
  });

  const [newSlot, setNewSlot] = useState<Partial<Slot>>({
    name: '',
    data_type: 'string',
    slot_description: '',
    slot_validation_policy: '',
    slot_validation_description: '',
    slot_inference_policy: 'infer_if_available'
  });

  const [newActivity, setNewActivity] = useState<Partial<Activity>>({
    name: '',
    type: 'response_activity',
    description: '',
    action_reference: '',
    required_slots: [],
    input_mapping: {},
    output_mapping: '',
    output_key: '',
    confirmation_policy: 'none'
  });

  // Testing state
  const [testInput, setTestInput] = useState('');
  const [isTestRunning, setIsTestRunning] = useState(false);
  const [testResult, setTestResult] = useState<any>(null);
  const [validationResult, setValidationResult] = useState('');



  const addSlot = () => {
    if (!newSlot.name) return;

    const slot: Slot = {
      id: `slot_${Date.now()}`,
      name: newSlot.name,
      data_type: newSlot.data_type || 'string',
      slot_description: newSlot.slot_description || '',
      slot_validation_policy: newSlot.slot_validation_policy || '',
      slot_validation_description: newSlot.slot_validation_description || '',
      slot_inference_policy: newSlot.slot_inference_policy || 'infer_if_available',
      resolver_strategy: newSlot.resolver_strategy
    };

    setEditData(prev => ({
      ...prev,
      slots: [...prev.slots, slot]
    }));

    setNewSlot({
      name: '',
      data_type: 'string',
      slot_description: '',
      slot_validation_policy: '',
      slot_validation_description: '',
      slot_inference_policy: 'infer_if_available'
    });
  };

  const removeSlot = (slotId: string) => {
    setEditData(prev => ({
      ...prev,
      slots: prev.slots.filter((s: Slot) => s.id !== slotId)
    }));
  };

  const addActivity = () => {
    if (!newActivity.name) return;

    const activity: Activity = {
      id: `activity_${Date.now()}`,
      name: newActivity.name,
      type: newActivity.type || 'response_activity',
      description: newActivity.description || '',
      action_reference: newActivity.action_reference || '',
      required_slots: newActivity.required_slots || [],
      input_mapping: newActivity.input_mapping || {},
      output_mapping: newActivity.output_mapping || '',
      output_key: newActivity.output_key || '',
      confirmation_policy: newActivity.confirmation_policy || 'none'
    };

    setEditData(prev => ({
      ...prev,
      activities: [...prev.activities, activity]
    }));

    setNewActivity({
      name: '',
      type: 'response_activity',
      description: '',
      action_reference: '',
      required_slots: [],
      input_mapping: {},
      output_mapping: '',
      output_key: '',
      confirmation_policy: 'none'
    });
  };

  const removeActivity = (activityId: string) => {
    setEditData(prev => ({
      ...prev,
      activities: prev.activities.filter((a: Activity) => a.id !== activityId)
    }));
  };

  const runQuickTest = async () => {
    if (!testInput.trim()) return;

    setIsTestRunning(true);
    setTestResult(null);

    try {
      const response = await fetch('http://localhost:8000/api/agent-studio/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          process_id: process?.id,
          test_input: testInput
        })
      });

      if (response.ok) {
        const result = await response.json();
        setTestResult(result);
      } else {
        setTestResult({
          success: false,
          response: 'Test failed - server error',
          steps: 0,
          execution_time: 0
        });
      }
    } catch (error) {
      setTestResult({
        success: false,
        response: 'Test failed - connection error',
        steps: 0,
        execution_time: 0
      });
    } finally {
      setIsTestRunning(false);
    }
  };

  const validateProcess = () => {
    const issues = [];

    if (!editData.name.trim()) issues.push('Process name is required');
    if (!editData.description.trim()) issues.push('Process description is required');
    if (editData.triggers.split(',').filter(t => t.trim()).length === 0) {
      issues.push('At least one trigger is required');
    }
    if (editData.slots.length === 0) issues.push('Consider adding slots to collect user input');
    if (editData.activities.length === 0) issues.push('At least one activity is required');

    if (issues.length === 0) {
      setValidationResult('✅ Process configuration is valid and ready for testing!');
    } else {
      setValidationResult(`⚠️ Issues found: ${issues.join(', ')}`);
    }
  };

  const handleSave = () => {
    const processData = {
      ...editData,
      triggers: editData.triggers.split(',').map(t => t.trim()).filter(t => t),
      keywords: editData.keywords.split(',').map(k => k.trim()).filter(k => k)
    };
    onSave(processData);
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Settings className="w-5 h-5" />
            Edit Process: {process?.name}
          </DialogTitle>
        </DialogHeader>

        <Tabs defaultValue="basic" className="space-y-4">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="basic">
              <Code className="w-4 h-4 mr-2" />
              Basic
            </TabsTrigger>
            <TabsTrigger value="slots">
              <Database className="w-4 h-4 mr-2" />
              Slots
            </TabsTrigger>
            <TabsTrigger value="activities">
              <Settings className="w-4 h-4 mr-2" />
              Activities
            </TabsTrigger>
            <TabsTrigger value="permissions">
              <Users className="w-4 h-4 mr-2" />
              Permissions
            </TabsTrigger>
            <TabsTrigger value="testing">
              <TestTube className="w-4 h-4 mr-2" />
              Testing
            </TabsTrigger>
          </TabsList>

          <TabsContent value="basic" className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="name">Process Name</Label>
                <Input
                  id="name"
                  value={editData.name}
                  onChange={(e) => setEditData(prev => ({ ...prev, name: e.target.value }))}
                  placeholder="Process name"
                />
              </div>
              <div>
                <Label htmlFor="status">Status</Label>
                <Select value={process?.status} disabled>
                  <SelectTrigger>
                    <SelectValue placeholder="Select status" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="draft">Draft</SelectItem>
                    <SelectItem value="testing">Testing</SelectItem>
                    <SelectItem value="published">Published</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div>
              <Label htmlFor="description">Description</Label>
              <Textarea
                id="description"
                value={editData.description}
                onChange={(e) => setEditData(prev => ({ ...prev, description: e.target.value }))}
                placeholder="Process description"
                rows={3}
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="triggers">Triggers (comma-separated)</Label>
                <Input
                  id="triggers"
                  value={editData.triggers}
                  onChange={(e) => setEditData(prev => ({ ...prev, triggers: e.target.value }))}
                  placeholder="reset password, forgot password"
                />
              </div>
              <div>
                <Label htmlFor="keywords">Keywords (comma-separated)</Label>
                <Input
                  id="keywords"
                  value={editData.keywords}
                  onChange={(e) => setEditData(prev => ({ ...prev, keywords: e.target.value }))}
                  placeholder="password, reset, help"
                />
              </div>
            </div>
          </TabsContent>

          <TabsContent value="slots" className="space-y-4">
            <div className="border rounded-lg p-4">
              <h4 className="font-medium mb-3">Add New Slot (Moveworks Style)</h4>
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <Label>Slot Name</Label>
                  <Input
                    placeholder="e.g., item_name, pto_type"
                    value={newSlot.name}
                    onChange={(e) => setNewSlot(prev => ({ ...prev, name: e.target.value }))}
                  />
                </div>
                <div>
                  <Label>Data Type</Label>
                  <Select
                    value={newSlot.data_type}
                    onValueChange={(value) => setNewSlot(prev => ({ ...prev, data_type: value as any }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="string">string</SelectItem>
                      <SelectItem value="number">number</SelectItem>
                      <SelectItem value="boolean">boolean</SelectItem>
                      <SelectItem value="custom_data_type">Custom Data Type</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="grid grid-cols-1 gap-4 mb-4">
                <div>
                  <Label>Slot Description</Label>
                  <Textarea
                    placeholder="e.g., The name of the item that the user wants to purchase"
                    value={newSlot.slot_description}
                    onChange={(e) => setNewSlot(prev => ({ ...prev, slot_description: e.target.value }))}
                    rows={2}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <Label>Validation Policy</Label>
                  <Input
                    placeholder="e.g., value > 0, value == TRUE"
                    value={newSlot.slot_validation_policy}
                    onChange={(e) => setNewSlot(prev => ({ ...prev, slot_validation_policy: e.target.value }))}
                  />
                </div>
                <div>
                  <Label>Inference Policy</Label>
                  <Select
                    value={newSlot.slot_inference_policy}
                    onValueChange={(value) => setNewSlot(prev => ({ ...prev, slot_inference_policy: value as any }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="infer_if_available">Infer slot value if available</SelectItem>
                      <SelectItem value="always_ask_explicitly">Always explicitly ask for slot</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="flex justify-end">
                <Button onClick={addSlot} size="sm">
                  <Plus className="w-4 h-4 mr-1" />
                  Add Slot
                </Button>
              </div>
            </div>

            <div className="space-y-2">
              <h4 className="font-medium">Configured Slots ({editData.slots.length})</h4>
              {editData.slots.map((slot: Slot) => (
                <div key={slot.id} className="border rounded p-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                      <Badge variant="outline">{slot.data_type}</Badge>
                      <div className="font-medium">{slot.name}</div>
                      {slot.slot_inference_policy === 'always_ask_explicitly' && (
                        <Badge variant="secondary">Always Ask</Badge>
                      )}
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeSlot(slot.id)}
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                  <div className="text-sm text-gray-600 mb-2">{slot.slot_description}</div>
                  {slot.slot_validation_policy && (
                    <div className="text-xs text-blue-600 bg-blue-50 px-2 py-1 rounded">
                      Validation: {slot.slot_validation_policy}
                    </div>
                  )}
                </div>
              ))}
              {editData.slots.length === 0 && (
                <div className="text-center py-8 text-gray-500">
                  No slots configured. Add slots to collect user information.
                </div>
              )}
            </div>
          </TabsContent>

          <TabsContent value="activities" className="space-y-4">
            <div className="border rounded-lg p-4">
              <h4 className="font-medium mb-3">Add New Activity (Moveworks Style)</h4>

              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <Label>Activity Name</Label>
                  <Input
                    placeholder="e.g., Submit Purchase Request"
                    value={newActivity.name}
                    onChange={(e) => setNewActivity(prev => ({ ...prev, name: e.target.value }))}
                  />
                </div>
                <div>
                  <Label>Activity Type</Label>
                  <Select
                    value={newActivity.type}
                    onValueChange={(value) => setNewActivity(prev => ({ ...prev, type: value as any }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="action_activity">Action Activity</SelectItem>
                      <SelectItem value="response_activity">Response Activity</SelectItem>
                      <SelectItem value="condition_activity">Condition Activity</SelectItem>
                      <SelectItem value="slot_collection_activity">Slot Collection Activity</SelectItem>
                      <SelectItem value="connector_activity">Connector Activity</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <Label>Action Reference</Label>
                  <Input
                    placeholder="e.g., mw.generate_text_action, firstname_lastname_submit_purchase_request"
                    value={newActivity.action_reference}
                    onChange={(e) => setNewActivity(prev => ({ ...prev, action_reference: e.target.value }))}
                  />
                </div>
                <div>
                  <Label>Output Key</Label>
                  <Input
                    placeholder="e.g., pr_classification, pto_balance_result"
                    value={newActivity.output_key}
                    onChange={(e) => setNewActivity(prev => ({ ...prev, output_key: e.target.value }))}
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 gap-4 mb-4">
                <div>
                  <Label>Description</Label>
                  <Textarea
                    placeholder="e.g., This Action Activity utilizes the built-in mw.generate_text_action to classify purchase requests"
                    value={newActivity.description}
                    onChange={(e) => setNewActivity(prev => ({ ...prev, description: e.target.value }))}
                    rows={2}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <Label>Output Mapping</Label>
                  <Input
                    placeholder="e.g., .openai_chat_completions_response.choices[0].message.content"
                    value={newActivity.output_mapping}
                    onChange={(e) => setNewActivity(prev => ({ ...prev, output_mapping: e.target.value }))}
                  />
                </div>
                <div>
                  <Label>Confirmation Policy</Label>
                  <Select
                    value={newActivity.confirmation_policy}
                    onValueChange={(value) => setNewActivity(prev => ({ ...prev, confirmation_policy: value as any }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="none">No confirmation required</SelectItem>
                      <SelectItem value="require_consent">Require consent from user</SelectItem>
                      <SelectItem value="custom">Custom confirmation</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="flex justify-end">
                <Button onClick={addActivity} size="sm">
                  <Plus className="w-4 h-4 mr-1" />
                  Add Activity
                </Button>
              </div>
            </div>

            <div className="space-y-2">
              <h4 className="font-medium">Configured Activities ({editData.activities.length})</h4>
              {editData.activities.map((activity: Activity) => (
                <div key={activity.id} className="border rounded p-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                      <Badge variant="outline">{activity.type.replace('_', ' ')}</Badge>
                      <div className="font-medium">{activity.name}</div>
                      {activity.confirmation_policy === 'require_consent' && (
                        <Badge variant="secondary">Requires Consent</Badge>
                      )}
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeActivity(activity.id)}
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                  <div className="text-sm text-gray-600 mb-2">{activity.description}</div>

                  <div className="grid grid-cols-2 gap-4 text-xs">
                    {activity.action_reference && (
                      <div className="bg-blue-50 px-2 py-1 rounded">
                        <span className="font-medium">Action:</span> {activity.action_reference}
                      </div>
                    )}
                    {activity.output_key && (
                      <div className="bg-green-50 px-2 py-1 rounded">
                        <span className="font-medium">Output Key:</span> {activity.output_key}
                      </div>
                    )}
                    {activity.output_mapping && (
                      <div className="bg-purple-50 px-2 py-1 rounded col-span-2">
                        <span className="font-medium">Output Mapping:</span> {activity.output_mapping}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {editData.activities.length === 0 && (
                <div className="text-center py-8 text-gray-500">
                  No activities configured. Add activities to define process behavior.
                </div>
              )}
            </div>
          </TabsContent>

          <TabsContent value="permissions" className="space-y-4">
            <div className="text-center py-8 text-gray-500">
              <Users className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <h3 className="text-lg font-medium mb-2">Permissions & Access Control</h3>
              <p>Configure user groups, roles, and access permissions for this process.</p>
              <p className="text-sm mt-2">Coming in next release...</p>
            </div>
          </TabsContent>

          <TabsContent value="testing" className="space-y-4">
            <div className="grid grid-cols-2 gap-6">
              <div className="space-y-4">
                <h4 className="font-medium">Quick Test</h4>
                <div className="border rounded-lg p-4">
                  <div className="space-y-3">
                    <Input
                      placeholder="Enter test message..."
                      value={testInput}
                      onChange={(e) => setTestInput(e.target.value)}
                    />
                    <Button
                      onClick={runQuickTest}
                      disabled={!testInput.trim() || isTestRunning}
                      className="w-full"
                    >
                      {isTestRunning ? (
                        <>
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                          Testing...
                        </>
                      ) : (
                        <>
                          <TestTube className="w-4 h-4 mr-2" />
                          Run Test
                        </>
                      )}
                    </Button>
                  </div>

                  {testResult && (
                    <div className="mt-4 p-3 bg-gray-50 rounded border">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium">Test Result</span>
                        <Badge variant={testResult.success ? "default" : "destructive"}>
                          {testResult.success ? "Success" : "Failed"}
                        </Badge>
                      </div>
                      <p className="text-sm text-gray-700 mb-2">{testResult.response}</p>
                      <div className="text-xs text-gray-500">
                        Steps: {testResult.steps} | Time: {testResult.execution_time}s
                      </div>
                    </div>
                  )}
                </div>
              </div>

              <div className="space-y-4">
                <h4 className="font-medium">Validation</h4>
                <div className="border rounded-lg p-4">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Slots Configuration</span>
                      <Badge variant="outline">
                        {editData.slots.length} configured
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Activities Configuration</span>
                      <Badge variant="outline">
                        {editData.activities.length} configured
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Triggers</span>
                      <Badge variant="outline">
                        {editData.triggers.split(',').filter(t => t.trim()).length} defined
                      </Badge>
                    </div>

                    <Button
                      onClick={validateProcess}
                      variant="outline"
                      className="w-full mt-4"
                    >
                      <Settings className="w-4 h-4 mr-2" />
                      Validate Configuration
                    </Button>

                    {validationResult && (
                      <div className="mt-3 p-3 bg-blue-50 rounded border border-blue-200">
                        <div className="text-sm">
                          <div className="font-medium text-blue-800 mb-1">Validation Result</div>
                          <div className="text-blue-700">{validationResult}</div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>

        <div className="flex justify-end gap-2 pt-4 border-t">
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={handleSave}>
            Save Process
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
