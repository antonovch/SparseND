
%% Initialization
rng(sum(clock*1e6),'twister')
thresh = 0.5; % average portion of zero elements in the generated data
infnan = 0.1; % average portion of either NaN or Inf elements

mdl = 20; % max length in a given dimension
maxdim = 5; % max number of dimensions of ND arrays
maxrep = 4; % max number if repetitions along a given dimension for repmat and repelem
numTests = 100; % number of runs to test each method with different random input

rscal = @() max(rand-thresh, 0)/ceil(rand-infnan); % random scalar generator
rvec_gen = @() max(rand(getRandVecSize(mdl))-thresh, 0); % random vector generator
rmat_gen = @() max(rand(randi(mdl,1,2))-thresh, 0); % random scalar generator
rtens_gen = @() max(rand(getRandTensSize(mdl,maxdim))-thresh, 0); % random scalar generator
rsz_gen = @(varargin) max(rand(varargin{:})-thresh, 0); % random array of specific size
insert_isnan = @(M) M./ceil(rand(size(M))-infnan); % random insertion of Infs and NaNs

rvec = @() insert_isnan(rvec_gen());
rmat = @() insert_isnan(rmat_gen());
rtens = @() insert_isnan(rtens_gen());
rsz = @(varargin) insert_isnan(rsz_gen(varargin{:}));

%% Note that isequal is overloaded to compare NaNs approproately.  
%%% Default behaviour is isequal(nan,nan) == false
%% Constructor

for n = 1:numTests
    for M = {rscal(), rvec(), rmat(), rtens()}
        assert(isequal(M{1},full(SparseND(M{1}))),...
            'Constructor failed for input size [%s].',num2str(size(M)))
    end
end

%% Indexing: referncing (subsref)

for n = 1:numTests
    for M = {rscal(), rvec(), rmat(), rtens()}
        M = M{1}; %#ok peel off the cell layer 
        N = ndims(M);
        S = SparseND(M);
        % linear indices
        subslin = arrayfun(@(n)randi(size(M,n),1,randi(size(M,n))), 1:N, 'UniformOutput', false);
        % randomly insert colon to mark all incidies, with some small-ish probability
        subslin(randi(N*5,1,N) <= N) = {':'};
        assert(isequal(M(subslin{:}), full(S(subslin{:}))),...
            'Linear index referencing failed.')
        
        % logical indices
        subslog = arrayfun(@(n)rand(1,size(M,n)) > 0.5, 1:N, 'UniformOutput', false);
        assert(isequal(M(subslog{:}), full(S(subslog{:}))),...
            'Logical index referencing failed.')
        
        % mixed indexing
        I = rand(1,N) > 0.5;
        subsmix = subslin;
        subsmix(I) = subslog(I);
        assert(isequal(M(subsmix{:}), full(S(subsmix{:}))),...
            'Mixed index referencing failed.')
        
    end
end

%% Indexing: assignment (subsasgn)

for n = 1:numTests
    for M = {rscal(), rvec(), rmat(), rtens()}
        F = M{1};  
        N = ndims(F);
        S = SparseND(F);
        % linear indices
        subslin = arrayfun(@(n)randi(size(M,n),1,randi(size(M,n))), 1:N, 'UniformOutput', false);
        % randomly insert colon to mark all incidies, with some small-ish probability
        subslin(randi(N*5,1,N) <= N) = {':'};
        sz = size(F(subslin{:}), 1:length(subslin)); % lazy way of getting the size of the assignment slice
        if n == 1
            % run once a few special cases:
            % 1/ assign zero
            F(subslin{:}) = 0;
            S(subslin{:}) = 0;
            assert(isequal(F, full(S)), 'Assignment of a zero failed.')
            % 2/ assign a scalar
            tmp = rscal();
            F(subslin{:}) = tmp;
            S(subslin{:}) = tmp;
            assert(isequal(F, full(S)), 'Scalar assignment failed.')
            % 3/ assign all zeros
            F(subslin{:}) = zeros(sz);
            S(subslin{:}) = zeros(sz);
            assert(isequal(F, full(S)), 'All-zero assignment failed.')
            if N < 3
                % 4/ assign all zero sparse
                F(subslin{:}) = sparse(sz(1),sz(2));
                S(subslin{:}) = sparse(sz(1),sz(2));
                assert(isequal(F, full(S)), 'All-zero sparse assignment failed.')
                % 5/ assign a sparse matrix
                tmp = sparse(max(rand(sz)-thresh, 0));
                F(subslin{:}) = tmp;
                S(subslin{:}) = tmp;
                assert(isequal(F, full(S)), 'All-zero assignment failed.')
            end
        end
        
        tmp = max(rand(sz)-thresh, 0);
        F(subslin{:}) = tmp;
        if mod(n,2) % odd n's -> tmp is full
             S(subslin{:}) = tmp;
        else % even n's -> tmp is sparse
            if N < 3
                S(subslin{:}) = sparse(tmp);
            else
                S(subslin{:}) = SparseND(tmp);
            end
        end
        
        assert(isequal(F, full(S)), 'Linear index assignment failed.')
        
        % logical indices
        subslog = arrayfun(@(n)rand(1,size(M,n)) > 0.5, 1:N, 'UniformOutput', false);
        szlog = cellfun(@sum, subslog);
        tmp = max(rand(szlog)-thresh, 0);
        F(subslog{:}) = tmp;
        if mod(n,2) % odd n's -> tmp is full
             S(subslog{:}) = tmp;
        else % even n's -> tmp is sparse
            if N < 3
                S(subslog{:}) = sparse(tmp);
            else
                S(subslog{:}) = SparseND(tmp);
            end
        end
        
        assert(isequal(F, full(S)), 'Logical index referencing failed.')
        
        % mixed indexing
        I = rand(1,N) > 0.5;
        subsmix = subslin;
        subsmix(I) = subslog(I);
        sz(I) = szlog(I);
        tmp = max(rand(sz)-thresh, 0);
        F(subsmix{:}) = tmp;
        if mod(n,2) % odd n's -> tmp is full
             S(subsmix{:}) = tmp;
        else % even n's -> tmp is sparse
            if N < 3
                S(subsmix{:}) = sparse(tmp);
            else
                S(subsmix{:}) = SparseND(tmp);
            end
        end
        
        assert(isequal(F, full(S)), 'Mixed index referencing failed.')
    end
end

%% reshape

for n = 1:numTests
    for M = {rscal(), rvec(), rmat(), rtens()}
        F = M{1}; 
        sz = size(F);
        N = ndims(F);
        S = SparseND(F);
        if n == 1
            % special case, same shape
            assert(isequal(F, full(reshape(S,size(S)))), 'Trivial transformation failed.')
        end
        for m = 1:randi(N) % number of attempted changes
            for k = randperm(10) % factor to decrease one dimension and increase another
                f = find(mod(sz,k)==0);
                if ~isempty(f)
                    % randomly choose a dimension
                    dim = f(randi(numel(f)));
                    % decrease its size k times
                    sz(dim) = sz(dim)/k;
                    % randomly choose some other dimension
                    dim2 = setdiff(1:N,dim);
                    dim2 = dim2(randi(numel(dim2)));
                    % increase its size by k
                    sz(dim2) = sz(dim2)*k;
                    break
                end
            end
        end
        
        F = reshape(F, sz);
        if rand < 0.5
            S = reshape(S, sz);
        else
            sz = num2cell(sz);
            sz{randi(N)} = [];
            S = reshape(S, sz{:});
        end
                
        assert(isequal(F, full(S)), 'reshape failed.')
        
    end
end

%% permute

for n = 1:numTests
    for M = {rscal(), rvec(), rmat(), rtens()}
        F = M{1}; 
        ord = randperm(ndims(F));
        S = SparseND(F);
        if n == 1
            % special case, same shape
            assert(isequal(F, full(permute(S,1:ndims(S)))), 'Trivial transformation failed.')
        end
        assert(isequal(permute(F,ord), full(permute(S,ord))), 'permute failed.')
    end
end

%% repmat

for n = 1:numTests
    for M = {rscal(), rvec(), rmat(), rtens()}
        F = M{1}; 
        r = randi(maxrep, 1, randi(ndims(F)));
        S = SparseND(F);
        assert(isequal(repmat(F,r), full(repmat(S,r))), 'repmat failed.')
    end
end

%% repelem

for n = 1:numTests
    for M = {rscal(), rvec(), rmat(), rtens()}
        F = M{1}; 
        S = SparseND(F);
        if isvector(F) && rand < 0.5
            r = {randi(maxrep)};
        else
            r = arrayfun(@(n)randi(maxrep, 1, size(F,n)),1:ndims(F),'UniformOutput',0);
        end
        assert(isequal(repelem(F,r{:}), full(repelem(S,r{:}))), 'repelem failed.')
    end
end

%% cat

for n = 1:numTests
    for M = {rscal(), rvec(), rmat(), rtens()}
        F = M{1}; 
        S = SparseND(F);
        dim = randi(ndims(F)+2);
        sz = size(F, 1:max(ndims(F),dim));
        B = cell(1, randi(5)); % up to 5 additional tensors to concatenate
        for m = 1:length(B)
            sz(dim) = randi(mdl);
            B{m} = rsz(sz);
        end
        F = cat(dim,F,B{:});
        assert(isequal(F, full(cat(dim,S,B{:}))), 'cat failed.')
    end
end

%% sort (implementation uses built-in sort, testing genericOverload and overloaded sparse)
%%% other optional arguments are taken care of automatically
for n = 1:numTests
    for M = {rscal(), rvec(), rmat(), rtens()}
        F = M{1}; 
        S = SparseND(F);
        dim = randi(ndims(F)+2);
        assert(isequal(sort(F,dim), full(sort(S,dim))), 'sort failed.')
    end
end

%% max (implementation uses built-in max, testing genericOverloadPair and overloaded sparse)
%%% not testing all possible inputs, as they should work automatically
for n = 1:numTests
    for M = {rscal(), rvec(), rmat(), rtens()}
        F = M{1}; 
        S = SparseND(F);
        if rand < 0.5
            dim = randi(ndims(F)+2);
            assert(isequal(max(F,[],dim), full(max(S,[],dim))), 'max failed with one array.')
        else
            B = rsz(size(F));
            assert(isequal(max(F,B), full(max(S,B))), 'max failed with two arrays.')
        end
    end
end

%% maxk (implementation uses built-in max, testing genericOverload and overloaded sparse)
%%% not testing all possible inputs, as they should work automatically
for n = 1:numTests
    for M = {rscal(), rvec(), rmat(), rtens()}
        F = M{1}; 
        S = SparseND(F);
        dim = randi(ndims(F)+2);
        k = randi(size(F,dim));
        assert(isequal(maxk(F,k,dim), full(maxk(S,k,dim))), 'maxk failed.')
    end
end

%% times (size compatibility is checked automatically)

for n = 1:numTests
    for M = {rscal(), rvec(), rmat(), rtens()}
        F = M{1}; 
        S = SparseND(F);
        B = rsz(size(F));
        assert(isequal(F.*B, full(S.*B)), 'times failed.')
    end
end

%%

function tf = isequal(a,b)
    ia = isnan(a);
    ib = isnan(b);
    a(ia) = 0;
    b(ib) = 0;
    tf = builtin('isequal',a,b) && builtin('isequal',ia,ib);
end

function sz = getRandVecSize(len)
    if nargin == 0
        len = 20;
    end
    if rand > 0.5
        sz = [1 randi(len)];
    else
        sz = [randi(len) 1];
    end
end

function sz = getRandTensSize(len, maxdim)
    if nargin < 2, maxdim = 5; end
    if nargin < 1, len = 20; end
    N = randi([3 maxdim]);
    sz = arrayfun(@(n)randi(len), 1:N);
end
