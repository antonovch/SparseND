classdef SparseND 
% Generalization of a sparse matrix for N dimensional numerical or logical
% tensors. The behavior of most of the methods intentionally mimics that 
% of Matlab's built-in sparse matrix class. 
% For instance, for a sparse ND-array S, cos(S) returns a 
% sparse matrix, even though it will be full, while addition of a
% non-sparse tensor returns a full one, even if you just add zero, etc.
% Many of the methods are implemented to convert back and forth to the
% built-in sparse class with appropriate reshaping and indexing for
% efficiency and simplicity.
%
% There are some of the (non-hidden) methods that do not exist for built-in
% sparse, or their behavior is modified (see help for more details): disp,
% head, sub, sparse, issquarem, reshape.
%
% Similar to the built-in sparse class, the idea was to create someting
% that behaves as close to the regular numerical/logical tensor as
% possible. With the number of dimensions being not fixed, there are only
% two ways to initialize a SparseND object:
% 1/ by passing a tensor (full or sparse) A, i.e. M = SparseND(A)
% 2/ by specifying indicies of non-zero elements in the corresponding 
% dimensions as column vectors, along with sizes, as separate arguemnts, 
% i.e. M = SparseND(i1,i2,i3...,v,dim1,dim2,dim3,...);
% For instance,
% A = rand(12,12,100); A(A<.8) = 0;
% subs = cell(1,ndims(A));
% [subs{:}] = ind2sub(size(A), find(A));
% sizeA = num2cell(size(A));
% M = SparseND(subs{:}, nonzeros(A), sizeA{:});
% N = SparseND(A);
% isequal(M,N) % true
%
% Note: overloded logical conversion returns sparse array (like the
% built-in). However, this prevents using SparseND as expression in if
% statements -- need to call the full method. This is not an issue for the 
% built-in sparse.
%
% by Anton Ovcharenko
% Stanford University, 2021

    properties (Access = public)
        sz (1,:) {mustBeInteger, mustBeNonnegative} % size vector of the tensor
        ind (:,1) {mustBeInteger, mustBePositive} % linear indicies vector of non-zero elements' locations
        data (:,1) {mustBeNumericOrLogical} % values in the corresponding slots, according to ind
    end
    
    methods

        function obj = SparseND(varargin)
        % Constructor. To avoid the possible ambiguities, only two use cases are supported:
        % 1/ S = SparseND(A), where A is a full matrix to be converted into a sparse one, S;
        % 2/ S = SparseND(i1,i2,i3...,v,dim1,dim2,dim3,...), where i's are vectors of indices
        % of non-zero values in the corresponding dimension, v - vector of those values, 
        % dim's - lengths of the corresponding dimension

            switch nargin
                case 0
                    error('SparseND:minrhs', 'Error using SparseND\nNot enough input arguments.')
                case 1
                    M = varargin{1};
                    assert(isnumeric(M) || islogical(M), 'Wrong data type: has to be numerical or logical.')
                    obj.sz = size(M);
                    obj.ind = find(M);
                    obj.data = nonzeros(M);
                otherwise
                    N = (nargin-1)/2;
                    assert(mod(N,1)==0, 'Number of input arguments has be to odd.')
                    obj.sz = [varargin{nargin-N+1:nargin}];
                    obj.ind = sub2ind(obj.sz, varargin{1:N});
                    obj.data = varargin{N+1};
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Basic operations
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function self = subsref(self, s)
        % As SparseND should behave as a numerical array, only ()-type 
        % referencing is overloaded, dot-type can be used to access some
        % of the methods. This can simplify notation, as commands can be
        % chained together, e.g. S(2,:).full, S.full(2,:), full(S(2,:))
        % all produce the same result.
            switch s(1).type
                case '()'
                    subs = s(1).subs;
                    if numel(subs) < numel(self.sz) 
                        % flatten remaining dimensions
                        self.sz = [self.sz(1:numel(subs)-1), prod(self.sz(numel(subs):end))];
                    end
                    % substitute all ':' with the full vectors and logical
                    % with indicies
                    for ii = 1:length(subs)
                        if isequal(subs{ii}, ':')
                            subs{ii} = 1:self.sz(ii);
                        elseif islogical(subs{ii})
                            subs{ii} = find(subs{ii});
                        end
                    end
                    
                    % get new size vector
                    sznew = cellfun(@numel, subs);
                    % get all combinations
                    [subs{:}] = ndgrid(subs{:});
                    % flatten to a vector
                    subs = cellfun(@(x)x(:), subs, 'UniformOutput', false);
                    % get all indicies
                    I = sub2ind(self.sz, subs{:});
                    % find which are nonzero
                    ii = intersect(self.ind, I);
                    % take into account possible duplicates (return a space
                    % logical matrix, as it can be large potentially)
                    indnew = any(sparse(ii(:)) == I(:).', 1);
                    if any(indnew)
                        % find data property elements, whose corresponding ind
                        % match required indicies
                        [indoldnew, ~] = find(self.ind == sparse(I(indnew)).');
                        % update self.data
                        self.data = self.data(indoldnew);
                    else
                        self.data = [];
                    end
                    % update self.ind 
                    self.ind = find(indnew);
                    % update the size vector
                    self.sz = sznew;
                case '.'
                    % If property requested, check access. This has to be done, 
                    % because otherwise overloaded subsref bypasses it's private/
                    % protected status and returns the result.
                    props = metaclass(self).PropertyList;
                    check = strcmp({props.Name}, s(1).subs);
                    if any(check) && ~strcmp(props(check).GetAccess, 'public')
                       error('No public property "%s" for class <a href="matlab:helpPopup SparseND">SparseND</a>.', s(1).subs) 
                    end
                    self = builtin('subsref', self, s(1));
                otherwise
                    error('%s-type referencing is not supported for SparseND class.', s(1).type)
            end
            if numel(s) > 1
                self = subsref(self, s(2:end));
            end
        end
        
        function self = subsasgn(self, s, v)
            switch s(1).type
                case '()'
                    assert(length(s) == 1, '()-indexing cannot be combined with other types for assignments.')
                    subs = s.subs;
                    SZ = self.sz;
                    if numel(subs) < numel(SZ) 
                        % flatten remaining dimensions
                        SZ = [self.sz(1:numel(subs)-1), prod(SZ(numel(subs):end))];
                        if isscalar(SZ), SZ = [SZ, 1]; end
                    end
                    
                    % substitute all ':' with the full vectors and logical
                    % with indicies
                    for ii = 1:length(subs)
                        if isequal(subs{ii}, ':')
                            subs{ii} = 1:SZ(ii);
                        elseif islogical(subs{ii})
                            subs{ii} = find(subs{ii});
                        end
                    end
                    slicesz = cellfun(@numel,subs);
                    
                    % get all combinations
                    [subs{:}] = ndgrid(subs{:});
                    % flatten to a vector
                    subs = cellfun(@(x)x(:), subs, 'UniformOutput', false);
                    % indices of all LHS elements 
                    I = sub2ind(SZ, subs{:});
                    
                    %%% Example using sparse; slower than manually
%                     sizev = size(v);
%                     if isscalar(v) || isequal(slicesz(slicesz~=1),sizev(sizev~=1))
%                         szold = size(self);
%                         self = sparse(self, ndims(self)+1);
%                         if isa(v, 'SparseND')
%                             v = sparse(v, ndims(v)+1);
%                         end
%                         self(I) = v(:);
%                         self = reshape(SparseND(self), szold);
%                     else
%                         error('size mismatch')
%                     end
                    
                    % indices of LHS non-zero elements 
                    [indnz, iindnz, lnzi] = intersect(self.ind, I);
                    % indices of LHS zero elements 
                    [IZ, lzi] = setdiff(I, self.ind);
                    
                    if isscalar(v)
                        % special case, separated for speed
                        v = full(v);
                        if v == 0
                            self.ind(iindnz) = [];
                            self.data(iindnz) = [];
                        else
                            [self.ind, II] = sort([self.ind; IZ]);
                            self.data(iindnz) = v;
                            self.data = [self.data; v*ones(numel(IZ),1)];
                            self.data = self.data(II);
                        end
                    else
                        % check sizes of the LHS and RHS
                        sizev = size(v);
                        if ~isequal(slicesz(slicesz>1),sizev(sizev>1))
                            str1 = repmat('%g-by-',1,numel(slicesz));
                            str1 = str1(1:end-4);
                            slicesz = num2cell(slicesz);
                            str2 = repmat('%g-by-',1,numel(sizev));
                            str2 = str2(1:end-4);
                            sizev = num2cell(sizev);
                            error('SparseND:subsassigndimmismatch',...
                                ['Unable to perform assignment because the size of ',...
                                'the left side is ',str1,' and the size of the right side is ',...
                                str2,'.'], slicesz{:}, sizev{:})
                        end
                        v = SparseND(v);
                        % 1/ values that are non-zero on both sides: overwrite
                        [~, ia, ib] = intersect(lnzi, v.ind);
                        self.data(iindnz(ia)) = v.data(ib);
                        % 2/ values that were zero: add
                        [~, ia, ib] = intersect(lzi, v.ind);
                        [self.ind, II] = sort([self.ind; IZ(ia)]);
                        self.data = [self.data; v.data(ib)];
                        self.data = self.data(II);
                        % 3/ values that become zero: remove
                        [~, ia] = setdiff(lnzi, v.ind);
                        [self.ind, ia] = setdiff(self.ind, indnz(ia));
                        self.data = self.data(ia);
                    end
                case '.' 
                    % if property requested, check access
                    props = metaclass(self).PropertyList;
                    check = strcmp({props.Name}, s(1).subs);
                    if any(check) && ~strcmp(props(check).SetAccess, 'public')
                       error('You cannot set the read-only property "%s" of <a href="matlab:helpPopup SparseND">SparseND</a>.', s(1).subs) 
                    end
                    self = builtin('subsasgn', self, s, v);
                otherwise
                    error('%s-type assigning is not supported for SparseND class.', s.type)
            end
        end
        
        function ind = subsindex(A)
            if islogical(A)
                ind = find(A) - 1;
            elseif nnz(A) < numel(A)
                error('SparseND:badsubscript','Array indices must be positive integers or logical values.')
            else
                ind = full(A) - 1;
            end
        end
        
        function ind = end(self, k, n)
            if k < n
                ind = self.sz(k);
            else
                ind = prod(self.sz(k:end));
            end
        end
        
        function B = colon(A, varargin)
            B = colon(A.data(1), varargin{:});
        end
        
        function self = set.sz(self, val)
        % This set. method makes sures that the size vector in the sz
        % property is written with no trailing ones and has at least two
        % elements (in case of scalars and empty vectors).
            if length(val) > 2
                % remove trailing 1s
                f = find(val ~= 1, 1, 'last');
                if ~isempty(f)
                    val = val(1:max(f,2));
                else
                    val = val(1:2);
                end
            elseif isscalar(val)
                val = [val 1];
            elseif isempty(val)
                val = [0 0];
            end
            self.sz = val;
        end
        
        function self = squeeze(self)
            self.sz = self.sz(self.sz ~= 1);
        end
        
        function varargout = size(self, varargin)
            q = [varargin{:}];
            if isempty(q), q = true(1,numel(self.sz)); end
            sizes = [self.sz, ones(1,max(q)-numel(self.sz))];
            if nargout < 2
                varargout = {sizes(q)};
            else
                varargout = num2cell(sizes(q));
            end
        end
        
        function l = length(self)
            l = max(self.sz);
        end
        
        function n = ndims(self)
            n = length(self.sz);
        end
        
        function n = nnz(self)
            n = numel(self.data);
        end
        
        function N = nonzeros(self)
            N = self.data;
        end
        
        function n = numel(self)
            n = prod(self.sz);
        end
        
        function n = numArgumentsFromSubscript(~,~,~)
        % Returns the number of SparseND instances. Must always return 1,
        % as it's impossible to have an array of SparseND objects. They
        % are to behave as numerical (or logical) tensors and contatentate 
        % data. In other words, builtin('numel', self) will always return 1.
            n = 1; 
        end
        
        function c = class(self)
        % Returns class of the data and not SparseND. To check for the
        % class, one should use isa(S,'SparseND').
            c = class(self.data);
        end
        
        function f = find(self)
            f = self.ind;
        end
        
        function disp(self, extended)
        % Overloaded display function that is called, most commnly, through
        % the command line, when you pass the SparseND variable name.
        % Unlike that of the built-in sparse class, this function does not 
        % show all the data by default, since ND arrays (even sparse) might 
        % have a lot. To have it shown, the second argument can be passed
        % as true to display the data. Note that in this case the message 
        % about the type and size of data is not shown.
            if nargin < 2 || nnz(self) == 0
                extended = false; % flip to true for default behavior same as built-in
                % sparse for double data type. Other types show message similar to this
                % function with extended=false, and then the data (extended=true).
            end
            if ~extended
                str = repmat('%dx',1,numel(self.sz));
                fprintf('  ')
                emptymsg = '';
                if numel(self) == 0
                    emptymsg = ' empty';
                elseif nnz(self) == 0
                    fprintf('All zero '); 
                end
                fprintf([str(1:end-1),emptymsg,' <a href="matlab:helpPopup SparseND">SparseND</a> ',...
                    '<a href="matlab:helpPopup %s">%s</a>', ' array\n'], ...
                    size(self), class(self), class(self));
                if nnz(self) ~= 0
                    fprintf('\t<a href="matlab:disp(%s,true)">Display data</a>\n\n',inputname(1))
                end
            else
                % all digits + commas + parentheses + 3 spaces before
                maxlen = sum(cellfun(@length,arrayfun(@num2str,size(self),'UniformOutput',false))) + ...
                    ndims(self) + 4;
                str = repmat('%d,',1,numel(self.sz)); 
                str = str(1:end-1);
                subs = sub(self);
                subs = [subs{:}];
                subs = num2cell(subs,2);
                for ii = 1:length(self.ind)
                    % separate step for location to tabulate them
                    loc_str = sprintf(['(', str, ')'], subs{ii});
                    fprintf('%*s\t %.4g\n', maxlen, loc_str, self.data(ii))
                end
                fprintf('\n')
            end
        end
        
        function head(self, N)
        % head(self, N) shows first N non-zero elements. Default N = 5.
            if nargin < 2, N = 5; end
            N = min(N, nnz(self));
            self.ind = self.ind(1:N);
            self.data = self.data(1:N);
            disp(self,true)
        end
        
        function varargout = sub(A, ndim)
        % Get subscripts of non-zero elements up to dimension dim. Note
        % that for ind2sub, if nargout < ndims(A), the returned subsctript
        % indicies correspond to an array where the last ndims(A)-nargout
        % dimensions flattened out. This is not how this function operates, 
        % however. This behavoir is acieved by passing the optional second 
        % argument ndim. While if the number of outputs of this function, N, is:
        %   - zero or one - all subscripts are returned as a cell array of vectors.
        %   - equal to ndims(A) or ndim, if given, -- subscripts are assigned accordinly
        %   - larger than one, but smaller than ndims(A) (or ndim if given) - 
        %     first N subscripts are assigned, i.e. not enough information to 
        %     index the data
            if nargin == 2
                subs = cell(1, ndim);
            else
                subs = cell(1, ndims(A));
            end
            [subs{:}] = ind2sub(A.sz, A.ind);
            if nargout < 2
                varargout = {subs};
            else
                varargout = subs;
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Matrix manipulation
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function S = sparse(S, dim)
        % Conversion to Matlab's sparse matrix. If S is not a matrix and dim is given, 
        % those dimensions will be flattened into columns, the remaining ones -- into rows.
        % If dim is empty or absent and S is an ND array, S is flattened into a column vector.
            dim_given = nargin == 2 && ~isempty(dim);
            if ~ismatrix(S) && ~dim_given
                S = reshape(S, [], 1);
            elseif dim_given
                N = max(ndims(S), max(dim));
                SZ = size(S,1:N);
                % move dimensions specified by dim to the end
                S = permute(S, [setdiff(1:N,dim), dim]);
                % reshape to a matrix (2D array)
                S = reshape(S, [], prod(SZ(dim)));
            end
            [i,j] = ind2sub(S.sz, S.ind);
            S = sparse(i,j,S.data,S.sz(1),S.sz(2));
        end
        
        function M = full(self)
        % Conversion to a full matrix.
            M = zeros(self.sz, class(self.data));
            M(self.ind) = self.data;
        end
        
        function self = reshape(self, varargin)
        % Overloaded default reshape function. New size can be defined
        % either as a vector of sizes, or individial arguments, one of
        % which may be [] to be determined automatically. Because of the
        % possible [] presence, we cannot run sznew = [varargin{:}] right
        % away. Unlike built-in reshape, this method allows to combine two ways of
        % new size definition, i.e. we can use not only reshape(A, 3, 2, 4, 5) or
        % reshape(A, [3,2,4,5]), but also reshape(A, [3,2], [4,5]) to do
        % the same.

            I = cellfun(@isempty, varargin);
            if any(I)
                assert(sum(I) == 1, 'SparseND:reshape:unknownDim', ...
                    'Error using reshape\nSize can only have one unknown dimension.')
                varargin{I} = prod(self.sz) / prod([varargin{~I}]);
                assert(mod(varargin{I},1)==0, 'SparseND:reshape:notDivisible',...
                    'Product of known dimensions, %g, not divisible into total number of elements, %g.',...
                    prod([varargin{~I}]), prod(self.sz))
            end
            sznew = [varargin{:}];
            assert(prod(sznew)==prod(self.sz), 'SparseND:reshape:notSameNumel', ...
                ['Number of elements must not change. Use [] as one of the size inputs ',...
                'to automatically calculate the appropriate size for that dimension.']);
            self.sz = sznew;
        end
        
        function self = permute(self, dimorder)
            assert(isequal(dimorder, unique(dimorder, 'stable')), ...
                'SparseND:permute:repeatedIndex', 'Error using permute\nDIMORDER cannot contain repeated permutation indices.')
            assert(isequal(1:numel(dimorder), sort(dimorder)), ...
                'SparseND:permute:tooFewIndices', 'Error using permute\nDIMORDER must have at least N elements for an N-D array.')
            subs = sub(self, max(ndims(self), numel(dimorder)));
            self.sz = size(self, dimorder);
            I = sub2ind(self.sz, subs{dimorder});
            [self.ind, II] = sort(I);
            self.data = self.data(II);
        end
        
        function self = repmat(self, varargin)
            n = [varargin{:}];
            if isscalar(n), n = [n n]; end
            if any(n==0)
                self.data = [];
                self.ind = [];
                sznew = size(self, max(ndims(self), length(n)));
                sznew(1:length(n)) = sznew(1:length(n)).*n;
                self.sz = sznew;
            end
            subs = sub(self, max(length(n), ndims(self)));
            for dim = 1:length(n)
                if n(dim) > 1
                    I = false(1,length(subs));
                    I(dim) = true;
                    subs{I} = reshape(subs{I}+(0:n(I)-1)*size(self,dim), [], 1);
                    subs(~I) = cellfun(@(x)repmat(x, n(I), 1), subs(~I), 'UniformOutput', false);
                end
            end
            sznew = size(self, 1:max(length(n),ndims(self)));
            sznew(1:length(n)) = sznew(1:length(n)).*n;
            self.sz = sznew;
            I = sub2ind(self.sz, subs{:});
            [self.ind, II] = sort(I);
            self.data = repmat(self.data, prod(n), 1);
            self.data = self.data(II);
        end

        function A = repelem(A, varargin)
            if isrow(A) && length(varargin) == 1
                varargin = [1, varargin];
            end
            % check correctness of input
            lens = cellfun(@length, varargin);
            check = lens == size(A, 1:numel(lens)) | lens == 1;
            assert(all(check), 'SparseND:repelem:mismatchedReplications',...
                ['Arguments must be scalars or vectors of ',...
                'length equal to the size in the corresponding dimension'])

            varargin = [varargin, repmat({1}, ndims(A)-length(varargin))];
            sznew = size(A, 1:max(ndims(A),length(lens)));
            I = cellfun(@isscalar, varargin);
            if any(I)
                sznew(I) = size(A,find(I)).*[varargin{I}]; 
            end
            sznew(~I) = cellfun(@sum, varargin(~I));
            
            % special case, zero dimension, early stopping
            if ~all(cellfun(@any, varargin))
                A.sz = sznew;
                A.data = [];
                A.ind = [];
                return
            end
            
            subs = sub(A, max(length(varargin), ndims(A)));
            A.sz = sznew; % has to be assigned after the use of sub method
            for dim = 1:length(varargin)
                if any(varargin{dim} ~= 1)
                    I = false(1,length(subs));
                    I(dim) = true;
                    % the tricky part here, we want to have a rectangular
                    % matrix, but each row can have a different number of
                    % shifts, so we pad with NaNs
                    copies = varargin{dim}(:)-1;
                    shifts = repmat(0:max(copies), size(copies,1), 1);
                    shifts(shifts > copies) = nan;
                    % shifts create neighbor duplicates, we also have to
                    % account for the cumulative shift due to all the
                    % elements before that were shifted for each dimension
                    if isscalar(copies)
                        cumshift = copies.*(subs{I}-1);
                    else
                        dimrep = cumsum(copies) - copies;
                        cumshift = dimrep(subs{I});
                        copies = copies(subs{I});
                        shifts = shifts(subs{I},:);
                    end
                    shifts = shifts + cumshift;
                    
                    subs{I} = subs{I}+shifts;
                    subs{I} = reshape(subs{I}.',[],1);
                    subs{I}(isnan(subs{I})) = [];
                    subs(~I) = cellfun(@(x)repelem(x, copies+1, 1), subs(~I), 'UniformOutput', false);
                    A.data = repelem(A.data, copies+1, 1);
                end
            end
            
            I = sub2ind(A.sz, subs{:});
            [A.ind, II] = sort(I);
            A.data = A.data(II);
        end
        
        function A = diag(A,k)
            if nargin == 1, k = 0; end
            if ismatrix(A)
                A = SparseND(diag(sparse(A),k));
            else
                error('SparseND:diag:firstInputMustBe2D', 'First input must be 2-D.')
            end
        end
        
        function A = circshift(A,K,dim)
        % Same as built-in, but when both K and dim are given, they don't
        % have to be scalars, just of same size, and shifts will be done in
        % order.
            if nargin < 3
                if isscalar(K)
                    dim = SparseND.parseInputDim(A.sz);
                else
                    dim = 1:length(K);
                end
            else
                [K, dim] = SparseND.repmatchsize(K, dim);
            end
            subs = sub(A);
            for n = 1:length(dim)
                subs{dim(n)} = mod(subs{dim(n)}+K(n), size(A,dim(n)));
                subs{dim(n)}(subs{dim(n)} == 0) = size(A,dim(n));
            end
            A.ind = sub2ind(A.sz, subs{:});
        end

        function A = flip(A, dim)
        % Unlike default flip, dim can be a vector, which leads to successive
        % applications of the function, i.e. flip(A, [4,3]) = flip(flip(A,4),3);
            if nargin < 2
                dim = SparseND.parseInputDim(A.sz);
            end
            subs = sub(A, max(ndims(A), max(dim)));
            for d = dim(:).'
                subs{d} = size(A,d) - subs{d} + 1;
            end
            A.ind = sub2ind(A.sz, subs{:});
            [A.ind, I] = sort(A.ind);
            A.data = A.data(I);
        end
        
        function A = flipud(A)
            A = flip(A,1);
        end
        
        function A = fliplr(A)
            A = flip(A,2);
        end
        
        function A = cat(dim, A, varargin)
            for i = 1:length(varargin)
                B = varargin{i};
                assert(isequalsize(A,B,dim), 'SparseND:catenate:dimensionMismatch', ...
                'Error using cat\nDimensions of arrays being concatenated are not consistent.')
                if ~isa(B, 'SparseND')
                    B = SparseND(B);
                end
                subsB = sub(B, max(dim,ndims(A)));
                subsB{dim} = subsB{dim} + size(A, dim);
                subsA = sub(A);

                % using temp var newsize in case dim > length(A.sz)
                newsize = size(A, 1:max(ndims(A), dim));
                newsize(dim) = newsize(dim) + size(B, dim);
                A.sz = newsize;

                % recalculate indices for A and calculate for B
                indA = sub2ind(A.sz, subsA{:});
                indB = sub2ind(A.sz, subsB{:});
                [A.ind, I] = sort([indA(:); indB(:)]);
                A.data = [A.data; B.data];
                A.data = A.data(I);
            end
        end

        function A = horzcat(A, varargin)
            A = cat(2, A, varargin{:});
        end

        function A = vertcat(A, varargin)
            A = cat(1, A, varargin{:});
        end

        function varargout = sort(A, varargin)
            varargout = cell(1, nargout);
            [dim, varargin] = SparseND.parseInputDim(A.sz, varargin, 'scalar');
            [varargout{:}] = SparseND.genericOverload('sort', A, dim, size(A, dim), 2, varargin{:});
        end
        
        function TF = issorted(A, varargin)
            [dim, varargin] = SparseND.parseInputDim(A.sz, varargin, 'scalar');
            TF = SparseND.genericOverload('issorted', A, dim, size(A, dim), 2, varargin{:});
        end

        function TF = issortedrows(A, varargin)
            assert(ismatrix(A), 'SparseND:issortedrows:MustBeMatrix', ...
                'Error using issortedrows\nFirst argument must be a vector or 2-dimensional matrix.')
            TF = issortedrows(sparse(A), varargin{:});
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Min/max operations
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function varargout = max(A, B, varargin)
            varargout = cell(1, nargout);
            if nargin == 1 || isempty(B)
                [dim, varargin] = SparseND.parseInputDim(A.sz, varargin);
                [varargout{:}] = SparseND.genericOverload('max', A, dim, 1, [], 2, varargin{:});
            else
                [varargout{:}] = SparseND.genericOverloadPair('max', A, B, varargin{:});
            end
        end

        function varargout = min(A, B, varargin)
            varargout = cell(1, nargout);
            if nargin == 1 || isempty(B)
                [dim, varargin] = SparseND.parseInputDim(A.sz, varargin);
                [varargout{:}] = SparseND.genericOverload('min', A, dim, 1, [], 2, varargin{:});
            else
                [varargout{:}] = SparseND.genericOverloadPair('min', A, B, varargin{:});
            end
        end
        
        function varargout = maxk(A,k,varargin)
            [dim, varargin] = SparseND.parseInputDim(A.sz, varargin);
            varargout = cell(1, nargout);
            [varargout{:}] = SparseND.genericOverload('maxk', A, dim, k, k, 2, varargin{:});
        end

        function varargout = mink(A,k,varargin)
            [dim, varargin] = SparseND.parseInputDim(A.sz, varargin);
            varargout = cell(1, nargout);
            [varargout{:}] = SparseND.genericOverload('mink', A, dim, k, k, 2, varargin{:});
        end      

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Logical operations
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function A = eq(A, B)
            A = SparseND.genericOverloadPair('eq', A, B);
        end
        function A = le(A, B)
            A = SparseND.genericOverloadPair('le', A, B);
        end
        function A = lt(A, B)
            A = SparseND.genericOverloadPair('lt', A, B);
        end
        function A = ge(A, B)
            A = SparseND.genericOverloadPair('ge', A, B);
        end
        function A = gt(A, B)
            A = SparseND.genericOverloadPair('gt', A, B);
        end
        function A = ne(A, B)
            A = SparseND.genericOverloadPair('ne', A, B);
        end
        function A = and(A, B)
            A = SparseND.genericOverloadPair('and', A, B);
        end
        function A = or(A, B)
            A = SparseND.genericOverloadPair('or', A, B);
        end
        function A = xor(A, B)
            A = SparseND.genericOverloadPair('xor', A, B);
        end
        function A = not(A)
            A = SparseND.genericOverload('not', A);
        end
        function A = all(A, varargin)
            dim = SparseND.parseInputDim(A.sz, varargin);
            A = SparseND.genericOverload('all', A, dim, 1, 2);
        end
        function A = any(A, varargin)
            dim = SparseND.parseInputDim(A.sz, varargin);
            A = SparseND.genericOverload('any', A, dim, 1, 2);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Element-wise arithmetic operations
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
        function A = times(A, B)
            [A, B] = SparseND.repmatchsize(A, B); 
            B = SparseND(B);
            % infs and nans in either A or B that correspond to zeros for 
            % the other, become nans and have to be added separately
            I = [setdiff(A.ind(isnan(A.data)|isinf(A.data)), B.ind);
                 setdiff(B.ind(isnan(B.data)|isinf(B.data)), A.ind)];
            [A.ind, ia, ib] = intersect(A.ind, B.ind);
            A.data = A.data(ia) .* B.data(ib);
            if ~isempty(I) 
                [A.ind, II] = sort([A.ind; I]);
                A.data = [A.data; NaN(numel(I),1)];
                A.data = A.data(II);
            end
        end
        
        function A = plus(A, B)
            
            % special case, separate for speed
            if isscalar(B)
                if ~issparse(B)
                    A = full(A) + B;
                else
                    if full(B) ~= 0
                        A = SparseND(full(A) + full(B));
                    end
                end
                return
            end
            A = SparseND.genericOverloadPair('plus', A, B);
        end

        function A = minus(A, B)
            
            % special case, separate for speed
            if isscalar(B)
                if ~issparse(B)
                    A = full(A) - B;
                else
                    if full(B) ~= 0
                        A = SparseND(full(A) - full(B));
                    end
                end
                return
            end
            A = SparseND.genericOverloadPair('minus', A, B);
        end

        function A = rdivide(A,B)
            A = SparseND.genericOverloadPair('rdivide', A, B);
        end
        
        function A = ldivide(B,A)
            A = SparseND.genericOverloadPair('ldivide', B, A);
        end
        
        function A = prod(A, varargin)
            [dim, varargin] = SparseND.parseInputDim(A.sz, varargin);
            A = SparseND.genericOverload('prod', A, dim, 1, 2, varargin{:});
        end

        function A = sum(A, varargin)
            [dim, varargin] = SparseND.parseInputDim(A.sz, varargin);
            A = SparseND.genericOverload('sum', A, dim, 1, 2, varargin{:});
        end

        function B = exp(A)
        % To mimic the behavior of build-in exp, has to return a sparse
        % matrix, even though it will be full, unless A has -infs.
            B = SparseND(ones(size(A)));
            B.data(A.ind) = exp(A.data);
            % check A, instead of B (as rmzeros would do), since it's
            % assumed nnz(A)<nnz(B)
            I = isinf(A.data) & A.data < 0; 
            if any(I)
                B.ind(A.ind(I)) = [];
                B.data(A.ind(I)) = [];
            end
        end
        
        function A = power(A,B)
            A = SparseND.genericOverload('power', A, [], [], B);
            % use genericOverload for handling of special cases involving infs and nans
        end
        function A = angle(A)
            A.data = angle(A.data);
            A = rmzeros(A);
        end
        function A = real(A)
            A.data = real(A.data);
            A = rmzeros(A);
        end
        function A = imag(A)
            A.data = imag(A.data);
            A = rmzeros(A);
        end
        function A = abs(A)
            A.data = abs(A.data);
        end
        function A = sqrt(A)
            A.data = sqrt(A.data);
        end
        function A = floor(A)
            A.data = floor(A.data);
            A = rmzeros(A);
        end
        function A = ceil(A)
            A.data = ceil(A.data);
            A = rmzeros(A);
        end
        function A = fix(A)
            A.data = fix(A.data);
            A = rmzeros(A);
        end
        function A = mod(A,m)
            A.data = mod(A.data,m);
            A = rmzeros(A);
        end
        function A = rem(A,b)
            A.data = rem(A.data,b);
            A = rmzeros(A);
        end
        function A = conj(A)
            A.data = conj(A.data);
        end
        function A = sign(A)
            A.data = sign(A.data);
        end
        function A = round(A,varargin)
            A.data = round(A.data,varargin{:});
            A = rmzeros(A);
        end
        function A = complex(A, B)
            A = A + 1i*B;
        end
        function A = log(A)
            A = SparseND.genericOverload('log', A);
        end
        function A = log10(A)
            A = SparseND.genericOverload('log10', A);
        end
        function A = log1p(A)
            A = SparseND.genericOverload('log1p', A);
        end
        function A = log2(A)
            A = SparseND.genericOverload('log2', A);
        end
        function A = uminus(A)
            A.data = uminus(A.data);
        end
        function A = uplus(A)
            A.data = uplus(A.data);
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Matrix operations
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % For many of these the requirement is the the tensors are matrices
        % or, even more narrowly, square matrices. For those, it makes more
        % sense to convert into Matlab's sparse matrix and use built-in
        % functions.
        
        function A = transpose(A)
            assert(ismatrix(A),'SparseND:transpose:NDArray','Transpose on ND array is not defined. Use PERMUTE instead.')
            A = SparseND(transpose(sparse(A))); 
        end
        
        function A = ctranspose(A)
            assert(ismatrix(A),'SparseND:ctranspose:NDArray','Transpose on ND array is not defined. Use PERMUTE instead.')
            A = SparseND(ctranspose(sparse(A))); 
        end
        
        function A = tril(A,varargin)
            assert(ismatrix(A),'SparseND:tril:firstInputMustBe2D','First input must be 2D.')
            A = SparseND(tril(sparse(A),varargin{:}));
        end
        
        function A = triu(A,varargin)
            assert(ismatrix(A),'SparseND:tril:firstInputMustBe2D','First input must be 2D.')
            A = SparseND(triu(sparse(A),varargin{:}));
        end
        
        function A = inv(A)
            assert(issquarem(A),'SparseND:inv:inputMustBeSquare','Error using inv\nInput matrix must be square.')
            A = SparseND(inv(sparse(A)));
        end
        
        function A = mldivide(A,B)
            assert(ismatrix(A) && ismatrix(B),'SparseND:mldivide:inputsMustBe2D',...
                'Arguments must be 2-D, or at least one argument must be scalar.')
            if isa(B,'SparseND')
                B = sparse(B);
            end
            A = SparseND(mldivide(sparse(A),B));
        end
        
        function A = mrdivide(B,A)
            assert(ismatrix(A) && ismatrix(B),'SparseND:mrdivide:inputsMustBe2D',...
                'Arguments must be 2-D, or at least one argument must be scalar.')
            if isa(A,'SparseND')
                A = sparse(A);
            end
            A = SparseND(mrdivide(sparse(B),A));
        end
        
        function A = mtimes(A,B)
            assert(ismatrix(A) && ismatrix(B),'SparseND:mtimes:inputsMustBe2D',...
                'Arguments must be 2-D, or at least one argument must be scalar.')
            if isa(B, 'SparseND')
                B = sparse(B);
            end
            A = SparseND(mldivide(sparse(A),B));
        end
        
        function d = det(A)
            assert(ismatrix(A),'SparseND:det:inputMustBe2D','Error using det\nInput must be 2-D.')
            d = det(sparse(A));
        end
        
        function A = expm(A)
            assert(issquarem(A),'SparseND:expm:inputMustBeSquare','Error using expm\nInput matrix must be square.')
            A = expm(sparse(A));
        end
        
        function A = expm1(A)
            assert(issquarem(A),'SparseND:expm1:inputMustBeSquare','Error using expm1\nInput matrix must be square.')
            A = SparseND(expm1(sparse(A)));
        end
        
        function varargout = sqrtm(A)
            assert(issquarem(A),'SparseND:sqrtm:inputMustBeSquare','Error using sqrtm\nInput matrix must be square.')
            varargout = cell(1,nargout);
            [varargout{:}] = sqrtm(full(A));
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Trigonometric operations
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function A = sin(A)
            A = SparseND.genericOverload('sin', A);
        end
        function A = cos(A)
            A = SparseND.genericOverload('cos', A);
        end
        function A = sec(A)
            A = SparseND.genericOverload('sec', A);
        end
        function A = csc(A)
            A = SparseND.genericOverload('csc', A);
        end
        function A = tan(A)
            A = SparseND.genericOverload('tan', A);
        end
        function A = cot(A)
            A = SparseND.genericOverload('cot', A);
        end
        function A = asin(A)
            A = SparseND.genericOverload('asin', A);
        end
        function A = acos(A)
            A = SparseND.genericOverload('acos', A);
        end
        function A = asec(A)
            A = SparseND.genericOverload('asec', A);
        end
        function A = acsc(A)
            A = SparseND.genericOverload('acsc', A);
        end
        function A = atan(A)
            A = SparseND.genericOverload('atan', A);
        end
        function A = acot(A)
            A = SparseND.genericOverload('acot', A);
        end
        function A = sind(A)
            A = SparseND.genericOverload('sind', A);
        end
        function A = cosd(A)
            A = SparseND.genericOverload('cosd', A);
        end
        function A = secd(A)
            A = SparseND.genericOverload('secs', A);
        end
        function A = cscd(A)
            A = SparseND.genericOverload('cscd', A);
        end
        function A = tand(A)
            A = SparseND.genericOverload('tand', A);
        end
        function A = cotd(A)
            A = SparseND.genericOverload('cotd', A);
        end
        function A = asind(A)
            A = SparseND.genericOverload('asind', A);
        end
        function A = acosd(A)
            A = SparseND.genericOverload('acosd', A);
        end
        function A = asecd(A)
            A = SparseND.genericOverload('asecd', A);
        end
        function A = acscd(A)
            A = SparseND.genericOverload('acscd', A);
        end
        function A = atand(A)
            A = SparseND.genericOverload('atand', A);
        end
        function A = acotd(A)
            A = SparseND.genericOverload('acotd', A);
        end
        function A = sinh(A)
            A = SparseND.genericOverload('sinh', A);
        end
        function A = cosh(A)
            A = SparseND.genericOverload('cosh', A);
        end
        function A = sech(A)
            A = SparseND.genericOverload('sech', A);
        end
        function A = csch(A)
            A = SparseND.genericOverload('csch', A);
        end
        function A = tanh(A)
            A = SparseND.genericOverload('tanh', A);
        end
        function A = coth(A)
            A = SparseND.genericOverload('coth', A);
        end
        function A = asinh(A)
            A = SparseND.genericOverload('asinh', A);
        end
        function A = acosh(A)
            A = SparseND.genericOverload('acosh', A);
        end
        function A = asech(A)
            A = SparseND.genericOverload('asech', A);
        end
        function A = acsch(A)
            A = SparseND.genericOverload('acsch', A);
        end
        function A = atanh(A)
            A = SparseND.genericOverload('atanh', A);
        end
        function A = acoth(A)
            A = SparseND.genericOverload('acoth', A);
        end
        function A = atan2(A,B)
            A = SparseND.genericOverloadPair('atan2', A, B);
        end
        function A = atan2d(A,B)
            A = SparseND.genericOverloadPair('atan2d', A, B);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Is-queries
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function tf = isempty(A)
            tf = isempty(A.data);
        end
        function B = isfinite(A)
            B = true(size(A));
            B(A.ind(~isfinite(A.data))) = false;
        end
        function A = isinf(A)
            A.ind = A.ind(isinf(A.data));
            A.data = true(size(A.ind));
        end
        function A = isnan(A)
            A.ind = A.ind(isnan(A.data));
            A.data = true(size(A.ind));
        end
        function tf = ismatrix(A)
            tf = numel(A.sz) == 2;
        end
        function tf = issquarem(A)
            tf = numel(A.sz) == 2 && A.sz(1) == A.sz(2);
        end
        function tf = isnumeric(A)
            tf = isnumeric(A.data);
        end
        function tf = islogical(A)
            tf = islogical(A.data);
        end
        function tf = isinteger(A)
            tf = isinteger(A.data);
        end
        function tf = isfloat(A)
            tf = isfloat(A.data);
        end
        function tf = isreal(A)
            tf = isreal(A.data);
        end
        function tf = isrow(A)
            tf = numel(A.sz) == 2 && A.sz(1) == 1;
        end
        function tf = iscolumn(A)
            tf = numel(A.sz) == 2 && A.sz(2) == 1;
        end
        function tf = isvector(A)
            tf = numel(A.sz) == 2 && any(A.sz == 1);
        end
        function tf = isscalar(A)
            tf = prod(A.sz) == 1;
        end
        function tf = issparse(~)
            tf = true;
        end
        function tf = istriu(A)
            if ismatrix(A)
                tf = istriu(sparse(A));
            else
                tf = false;
            end
        end
        function tf = istril(A)
            if ismatrix(A)
                tf = istril(sparse(A));
            else
                tf = false;
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Type conversion methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function A = int8(A)
            A.data = int8(A.data);
        end
        function A = int16(A)
            A.data = int16(A.data);
        end
        function A = int32(A)
            A.data = int32(A.data);
        end
        function A = int64(A)
            A.data = int64(A.data);
        end
        function A = uint8(A)
            A.data = uint8(A.data);
        end
        function A = uint16(A)
            A.data = uint16(A.data);
        end
        function A = uint32(A)
            A.data = uint32(A.data);
        end
        function A = uint64(A)
            A.data = uint64(A.data);
        end
        function A = single(A)
            A.data = single(A.data);
        end
        function A = double(A)
            A.data = double(A.data);
        end
        function A = logical(A)
            A.data = logical(A.data);
        end
               
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Other
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function A = diff(A, n, dim)
            if nargin < 3
                dim = parseInputDim(A.sz, {}, 'scalar');
            end
            if nargin < 2 || isempty(n), n = 1; end
            A = SparseND.genericOverload('diff', A, dim, size(A,dim)-n, n, 2);
        end

    end
    
    methods (Hidden = true)

        function tf = isequalsize(A, B, ignoredim)
        % Checks if two ND arrays A and B have equal size, except in those
        % dimensions specified by the optional ignoredim vector.
            if nargin == 2, ignoredim = []; end
            N = max([ndims(A), ndims(B), ignoredim]);
            szA = size(A, 1:N);
            szB = size(B, 1:N);
            szA(ignoredim) = [];
            szB(ignoredim) = [];
            tf = isequal(szA, szB);
        end
        
        function self = rmzeros(self)
        % removes zeros that might be present in self SparseND object
            I = self.data == 0;
            self.data(I) = [];
            self.ind(I) = [];
        end
        
    end

    methods (Hidden, Static)

        function varargout = repmatchsize(A, B)
        % Checks compatibility of the matrices and 
        % if nargout == 2: replicates dimensions of A and B for their sizes to be equal and returns them
        % if nargout == 1: returns true if A and B have compatible dimensions, false otherwise
        % if nargout == 0: checks for compatibility with assert (error if they're uncompatble)
        % Does not make A or B sparse, if they are not already.
            szA = size(A);
            szB = size(B);
            dimdiff = numel(szA) - numel(szB);
            if dimdiff < 0
                szA = [szA, ones(1,-dimdiff)];
            else
                szB = [szB, ones(1,dimdiff)];
            end
            test = all(szA == szB | szA == 1 | szB == 1);
            if nargout == 1
                varargout = {test};
            else
                assert(test, 'SparseND:dimagree', 'Matrix dimensions must agree.')
                if nargout == 0
                    varargout = {};
                else
                    sznew = max(szA, szB);
                    A = repmat(A, sznew.^(szA < szB));
                    B = repmat(B, sznew.^(szA > szB));
                    varargout = {A, B};
                end
            end
        end

        function [dim, args] = parseInputDim(sz, args, varargin)
        % This is a common routine, due to a typical sintax imput of Matlab.
        % Namely, after the operands, the first argument is often dimension 
        % (scalar or vector) or 'all' to indicate all the dimensions.
            if nargin == 1, args = {}; end
            if ~isempty(args) && isnumeric(args{1})
                dim = args{1};
                args(1) = [];
            elseif ~isempty(args) && strcmp(args{1},'all')
                dim = 1:numel(sz);
                args(1) = [];
            else
                dim = find(sz ~= 1, 1, 'first');
                if isempty(dim)
                    dim = 1;
                end
            end
            for i = 1:length(varargin)
                switch varargin{i}
                    case 'scalar'
                        assert(isscalar(dim) && dim > 0 && mod(dim,1)==0, ...
                        'SparseND:notPosInt', ...
                        'Error using sort\nDimension argument must be a positive integer.')
                end
            end
        end

        function varargout = genericOverload(fun, A, dim, k, varargin)
        % Generic overload for built-in functions that operate on sparse matrix A with  
        % potentially other arguments in varargin. For functions that change the size of A,
        % i.e. sum along dimension dim (optional), so that size(A,dim) == k in the result.
            varargout = cell(1, nargout);
            if nargin < 4 || isempty(k), k = 1; end
            if nargin < 3, dim = []; end
            if isempty(dim) && ~ismatrix(A)
                dim = ndims(A) + 1;
            end
            if ~isscalar(dim) && isscalar(k)
                k = repelem(k, length(dim));
            elseif ~isequal(size(dim), size(k))
                error('Sizes of dim and k must match, or k must be a scalar.')
            end
            newsize = size(A, 1:max(ndims(A), max(dim)));
            if any(k ~= 1)
                newsize(dim) = [];
                newsize = [newsize, k];
            else
                newsize(dim) = k;
            end
            newnumel = prod(newsize);
            N = length(newsize);
            [varargout{:}] = feval(fun, sparse(A,dim), varargin{:});
            for i = 1:nargout
                if numel(varargout{i}) == newnumel
                    varargout{i} = reshape(SparseND(varargout{i}), newsize);
                    if any(k ~= 1)
                        I = false(1, N);
                        I(dim) = true;
                        order = 1:N;
                        order(I) = N-length(dim)+1:N;
                        order(~I) = 1:N-length(dim);
                        varargout{i} = permute(varargout{i}, order);
                    end
                end
            end
        end

        function varargout = genericOverloadPair(fun, A, B, varargin)
        % Generic overload for built-in pair element-wise operations. Might not be the most
        % time and memory-efficient, but a simple enough way of doing things.  
        % 'fun' is a function name or handle that accepts two (sparse) numerical matrices as
        % first arguments. The reason why genericOverload cannot be used for all cases is that
        % since sparse(), as it is overloaded here, flattens N-D tensors into vectors, A and B
        % have to have the same size (unless they are matrices).
            if (~ismatrix(A) || ~ismatrix(B)) && ~(isscalar(A) || isscalar(B))
                % if at least one is ND array and neither is scalar
                [A, B] = SparseND.repmatchsize(A, B);
            end
            if ~ismatrix(B) && ~isa(B,'SparseND')
                B = SparseND(B);
            end
            varargout = cell(1, nargout);
            [varargout{:}] = feval(fun, sparse(A), sparse(B), varargin{:});
            for i = 1:nargout
                if numel(varargout{i}) == numel(A)
                    if issparse(varargout{i})
                        varargout{i} = SparseND(varargout{i});
                    end
                    varargout{i} = reshape(varargout{i}, size(A));
                end
            end
        end
    end
    
end