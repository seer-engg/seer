import { Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface FormButtonsProps {
  onTest: () => void;
  isTesting: boolean;
  isBinding: boolean;
  isEditMode: boolean;
}

export function FormButtons({ onTest, isTesting, isBinding, isEditMode }: FormButtonsProps) {
  return (
    <div className="flex justify-end gap-2 pt-2">
      <Button
        type="button"
        variant="outline"
        onClick={onTest}
        disabled={isTesting || isBinding}
      >
        {isTesting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
        Test Connection
      </Button>
      <Button type="submit" disabled={isTesting || isBinding}>
        {isBinding && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
        {isEditMode ? 'Update Database' : 'Save'}
      </Button>
    </div>
  );
}
