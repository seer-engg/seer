import { useState, useEffect } from 'react';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';

interface FormConfigEditorProps {
  initialConfig?: {
    title?: string;
    description?: string;
    submitButtonText?: string;
    successMessage?: string;
  };
  onChange: (config: Record<string, string>) => void;
}

export const FormConfigEditor: React.FC<FormConfigEditorProps> = ({
  initialConfig = {},
  onChange,
}) => {
  const [title, setTitle] = useState(initialConfig.title || 'Form');
  const [description, setDescription] = useState(initialConfig.description || '');
  const [submitButtonText, setSubmitButtonText] = useState(initialConfig.submitButtonText || 'Submit');
  const [successMessage, setSuccessMessage] = useState(
    initialConfig.successMessage || 'Thank you for your submission!'
  );

  // Debounced update
  useEffect(() => {
    const timer = setTimeout(() => {
      onChange({
        title,
        description,
        submitButtonText,
        successMessage,
      });
    }, 500);

    return () => clearTimeout(timer);
  }, [title, description, submitButtonText, successMessage, onChange]);

  return (
    <div className="space-y-4 p-4 border rounded-md bg-muted/20">
      <h4 className="text-sm font-semibold">Form Settings</h4>

      <div className="space-y-2">
        <Label htmlFor="form-title">Form Title</Label>
        <Input id="form-title"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Contact Us"
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="form-description">Description</Label>
        <Textarea id="form-description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Get in touch with our team..." rows={2}
        />
      </div>

      <div className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="submit-button">Submit Button</Label>
          <Input id="submit-button"
            value={submitButtonText}
            onChange={(e) => setSubmitButtonText(e.target.value)} placeholder="Submit"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="success-message">Success Message</Label>
          <Input id="success-message"
            value={successMessage}
            onChange={(e) => setSuccessMessage(e.target.value)} placeholder="Thank you!"
          />
        </div>
      </div>
    </div>
  );
};
