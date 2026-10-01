function r = any(x, varargin)
  % Octave lacks any(x, 'all'); FSDA's FSM uses it.  Same result as MATLAB's any(x, 'all').
  if nargin == 2 && ischar(varargin{1}) && strcmpi(varargin{1}, 'all')
    r = builtin('any', x(:));
  else
    r = builtin('any', x, varargin{:});
  end
end
