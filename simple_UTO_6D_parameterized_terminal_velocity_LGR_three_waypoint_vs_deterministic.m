function results = simple_UTO_6D_parameterized_terminal_velocity_LGR_three_waypoint_vs_deterministic()
% SIMPLE_UTO_6D_PARAMETERIZED_TERMINAL_VELOCITY_LGR_THREE_WAYPOINT_VS_DETERMINISTIC
% The physical state remains 9-D: [position; velocity; attitude].
% Initial position and attitude are uncertain, giving a 6-D random vector
% and seven simplex sigma trajectories. Three independent FAST-LIO belief
% presets and waypoint parameters match the Standard TO vs Z2 source file.
% One parameterized UTO template and one parameterized Standard TO template
% are reused by all three waypoints. Waypoint 3 uses a terminal mean-velocity
% tolerance instead of a hard zero equality. Each solve is cold-started;
% no validation is performed.
% Returned controls, states, means, and covariances use physical units.

import casadi.*

%% 1. Shared problem settings
P.nRegions = 2;
P.nodesPerRegion = 5;
P.nControlConstraintPointsPerRegion = 31;
P.Tmin=0.5;P.Tmax=10;P.horizonBuffer=0.30;
P.finalStopBuffer=0.20;P.cruiseSpeed=1;

P.g = 9.81;
P.tauAtt = 0.35;
P.drag = 0.12;
P.uMin = [0;-0.48;-0.48;-1.20];
P.uMax = [18;0.48;0.48;1.20];
P.controlInteriorMargin = [0.05;0.005;0.005;0.005];
P.uHover = [P.g;0;0;0];
P.vMax = 4;
P.angleMax = 0.60;

% Manual NLP scaling. Physical variables are x=Sx*z and u=Su*v.
P.xScale = [3;1;1.20;4;4;4;0.60;0.60;0.60];
P.uScale = [P.g;0.48;0.48;1.20];
P.Sx = diag(P.xScale);
P.SxInv = diag(1./P.xScale);
P.Su = diag(P.uScale);

P.Qposition = diag([10,10,10]);
P.Qe = diag([10,10,10,1,1,1]);
P.QterminalVelocity = eye(3);
P.terminalVelocityWeight = 7e-3;
P.finalVelocityWeight = 0.10;
P.finalVelocityTolerance = 0.05; % component-wise mean velocity bound [m/s]
P.Ru = diag([2e-3,0.15,0.15,0.04]);
P.Rdu = diag([1e-2,0.30,0.30,0.08]);
P.controlEffortWeight = 1e-6;
P.controlSmoothnessWeight = 1e-6;

P.ipoptTol = 2e-5;
P.ipoptAcceptableTol = 1e-4;
P.ipoptMaxIter = 900;

[tau,D] = lgrOperators(P.nodesPerRegion);
[unitSimplex,w] = simplexSigmaPoints(6);
nSigma = numel(w);
[means,P0UncertainAll,waypoints,velocityReferences]=threeWaypointMission(P);
uncertainStateIndices=[1 2 3 7 8 9];
uncertaintyMap=zeros(9,6);uncertaintyMap(uncertainStateIndices,:)=eye(6);

%% 2. One-time parameterized NLP construction
setup=struct();
buildClock=tic;
standardNLP=buildStandardTOProblem(P,tau,D);
setup.standard=toc(buildClock);
buildClock=tic;
utoNLP=buildUTOProblem(P,tau,D,w,nSigma);
setup.UTO=toc(buildClock);

fprintf('\nOne-time prebuild of shared parameterized NLP templates\n');
fprintf('UTO template build      %10.6f s\n',setup.UTO);
fprintf('Standard template build %10.6f s\n',setup.standard);

legResults=repmat(struct(),1,3);
for leg=1:3
isFinal=leg==3;
x0Mean=means(:,leg);pGoal=waypoints(:,leg);
vRef=velocityReferences(:,leg);
P.tf=presetHorizon(x0Mean,pGoal,isFinal,P);
P0Uncertain=P0UncertainAll(:,:,leg);
P0=uncertaintyMap*P0Uncertain*uncertaintyMap.';
X0Sigma=repmat(x0Mean,1,nSigma)+ ...
    uncertaintyMap*chol(P0Uncertain,'lower')*unitSimplex;
X0Sigma(7:9,:)=wrapAngles(X0Sigma(7:9,:));

UNLP=utoNLP;
DNLP=standardNLP;
if leg==1
    % Charge the one-time shared-template build only to its first use.
    time.UTO.build=setup.UTO;
    time.standardTO.build=setup.standard;
else
    time.UTO.build=0;
    time.standardTO.build=0;
end
if isFinal
    terminalMode=1;
    terminalVelocityLower=-P.finalVelocityTolerance*ones(3,1);
    terminalVelocityUpper= P.finalVelocityTolerance*ones(3,1);
else
    terminalMode=0;
    terminalVelocityLower=-P.vMax*ones(3,1);
    terminalVelocityUpper= P.vMax*ones(3,1);
end
sizes.UTO=[UNLP.opti.nx,UNLP.opti.ng];
sizes.standardTO=[DNLP.opti.nx,DNLP.opti.ng];

%% 3. Cold-start UTO solve using a prebuilt template
utoTotalTimer = tic;
setUTOParameters(UNLP,X0Sigma,pGoal,vRef,P.tf,terminalMode, ...
    terminalVelocityLower,terminalVelocityUpper);
setInitialGuess(UNLP.opti,UNLP.U,UNLP.X,X0Sigma,pGoal,vRef,P.tf,tau,P);

utoSolveTimer = tic;
solU = UNLP.opti.solve();
time.UTO.solve = toc(utoSolveTimer);
statsU = solU.stats();
time.UTO.ipopt = getSolverWallTime(statsU);
time.UTO.iterations = getIterationCount(statsU);
[Uuto,Xuto] = extractLGRSolution(solU,UNLP.U,UNLP.X,P);
time.UTO.onlineTotal = toc(utoTotalTimer);
time.UTO.interfaceAndExtraction = max(0,time.UTO.onlineTotal-time.UTO.solve);
time.UTO.coldTotal=time.UTO.build+time.UTO.onlineTotal;

%% 4. Standard TO solve using its corresponding prebuilt template
detTotalTimer = tic;
setStandardParameters(DNLP,x0Mean,pGoal,vRef,P.tf,terminalMode, ...
    terminalVelocityLower,terminalVelocityUpper);
setInitialGuess(DNLP.opti,DNLP.U,DNLP.X,x0Mean,pGoal,vRef,P.tf,tau,P);

detSolveTimer = tic;
solD = DNLP.opti.solve();
time.standardTO.solve = toc(detSolveTimer);
statsD = solD.stats();
time.standardTO.ipopt = getSolverWallTime(statsD);
time.standardTO.iterations = getIterationCount(statsD);
[Udet,Xdet] = extractLGRSolution(solD,DNLP.U,DNLP.X,P);
time.standardTO.onlineTotal = toc(detTotalTimer);
time.standardTO.interfaceAndExtraction = max(0, ...
    time.standardTO.onlineTotal-time.standardTO.solve);
time.standardTO.coldTotal=time.standardTO.build+time.standardTO.onlineTotal;

%% 5. Terminal UT comparison only; no Monte Carlo validation
postTimer = tic;
terminalU = squeeze(Xuto(:,end,:,P.nRegions));
if nSigma==1,terminalU=terminalU(:);end
terminalD = zeros(9,nSigma);
for i=1:nSigma
    terminalD(:,i) = propagateLGRControlTerminal( ...
        X0Sigma(:,i),Udet,tau,P.tf,P);
end
[meanU,covU] = weightedTerminalStatistics(terminalU,w);
[meanD,covD] = weightedTerminalStatistics(terminalD,w);
time.comparisonPostprocess = toc(postTimer);

fprintf('\nWaypoint %d/3: parameterized-velocity 6-D UTO versus Standard TO\n',leg);
fprintf('Initial uncertainty: position + attitude; preset FAST-LIO belief\n');
fprintf('Manual state/control scaling enabled; IPOPT internal scaling disabled\n');
fprintf('Horizon %.6f s; final-stop leg: %d\n',P.tf,isFinal);
fprintf('Terminal mean-velocity bounds: [% .3f,% .3f] m/s per component\n', ...
    terminalVelocityLower(1),terminalVelocityUpper(1));
fprintf('NLP sizes [variables constraints], UTO [%d %d], Standard [%d %d]\n', ...
    sizes.UTO,sizes.standardTO);
fprintf('Sigma trajectories: %d; control checks/region: %d\n', ...
    nSigma,P.nControlConstraintPointsPerRegion);
fprintf('Target                 : [% .6f % .6f % .6f]^T\n',pGoal);
fprintf('UTO terminal mean      : [% .6f % .6f % .6f]^T\n',meanU(1:3));
fprintf('Standard terminal mean : [% .6f % .6f % .6f]^T\n',meanD(1:3));
fprintf('UTO terminal mean speed: %.6e\n',norm(meanU(4:6)-vRef));
fprintf('TO terminal mean speed : %.6e\n',norm(meanD(4:6)-vRef));
fprintf('trace(P_position,f), UTO     : %.6e\n',trace(covU(1:3,1:3)));
fprintf('trace(P_position,f), standard: %.6e\n',trace(covD(1:3,1:3)));
fprintf('UTO costs [cov vel effort smooth]: %.6e %.6e %.6e %.6e\n', ...
    solU.value(UNLP.Jcov),solU.value(UNLP.Jvelocity), ...
    P.controlEffortWeight*solU.value(UNLP.Jeffort), ...
    P.controlSmoothnessWeight*solU.value(UNLP.Jsmooth));
fprintf('TO costs  [vel effort smooth]    : %.6e %.6e %.6e\n', ...
    solD.value(DNLP.Jvelocity),P.controlEffortWeight*solD.value(DNLP.Jeffort), ...
    P.controlSmoothnessWeight*solD.value(DNLP.Jsmooth));

fprintf('\nReal-machine compute time (seconds; reusable NLP templates)\n');
fprintf('                         Cold UTO     Standard TO\n');
fprintf('Template build charge%10.6f    %10.6f\n', ...
    time.UTO.build,time.standardTO.build);
fprintf('IPOPT solve           %10.6f    %10.6f\n', ...
    time.UTO.solve,time.standardTO.solve);
fprintf('IPOPT iterations      %10d    %10d\n', ...
    time.UTO.iterations,time.standardTO.iterations);
fprintf('Interface/extraction  %10.6f    %10.6f\n', ...
    time.UTO.interfaceAndExtraction,time.standardTO.interfaceAndExtraction);
fprintf('ONLINE PREBUILT TIME  %10.6f    %10.6f\n', ...
    time.UTO.onlineTotal,time.standardTO.onlineTotal);
fprintf('COLD ALLOCATED TOTAL  %10.6f    %10.6f\n', ...
    time.UTO.coldTotal,time.standardTO.coldTotal);
fprintf('Comparison postproc.  %10.6f    (excluded from onboard time)\n', ...
    time.comparisonPostprocess);

%% 5. Terminal covariance plots
figure('Color','w','Name',sprintf('Waypoint %d terminal covariance',leg));
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
planes={[1 2],[1 3],[2 3]};
names={'XY plane','XZ plane','YZ plane'};
labels={{'X (m)','Y (m)'},{'X (m)','Z (m)'},{'Y (m)','Z (m)'}};
for j=1:3
    dims=planes{j};nexttile;hold on;
    hD=plotCovarianceEllipse(meanD(1:3),covD(1:3,1:3),dims,[1 0 0],':','o');
    hU=plotCovarianceEllipse(meanU(1:3),covU(1:3,1:3),dims,[0 0.5 0],'-','none');
    hMD=plot(meanD(dims(1)),meanD(dims(2)),'ro','MarkerFaceColor','r');
    hMU=plot(meanU(dims(1)),meanU(dims(2)),'gs','MarkerFaceColor',[0 0.5 0]);
    hG=plot(pGoal(dims(1)),pGoal(dims(2)),'k+','LineWidth',2,'MarkerSize',10);
    axis equal;grid on;xlabel(labels{j}{1});ylabel(labels{j}{2});
    title(sprintf('WP%d %s - terminal 95%% covariance',leg,names{j}));
    legend([hD,hU,hMD,hMU,hG],{'Standard TO covariance','UTO covariance', ...
        'Standard TO mean','UTO mean','Goal'},'Location','best');
end

%% 6. Store this leg
legResults(leg).parameters=struct('settings',P,'x0Mean',x0Mean, ...
    'initialCovariance',P0,'initialUncertainCovariance',P0Uncertain, ...
    'uncertainStateIndices',uncertainStateIndices, ...
    'initialSigmaStates',X0Sigma,'weights',w, ...
    'pGoal',pGoal,'vRef',vRef,'tau',tau,'D',D, ...
    'stateScale',P.xScale,'controlScale',P.uScale,'isFinal',isFinal, ...
    'terminalMode',terminalMode, ...
    'terminalVelocityLower',terminalVelocityLower, ...
    'terminalVelocityUpper',terminalVelocityUpper);
legResults(leg).problemSizes=sizes;
legResults(leg).UTO=struct('control',Uuto,'sigmaStates',Xuto, ...
    'terminalMean',meanU,'terminalCovariance',covU, ...
    'objective',solU.value(UNLP.objective));
legResults(leg).standardTO=struct('control',Udet,'nominalStates',Xdet, ...
    'terminalSigmaStates',terminalD,'terminalMean',meanD, ...
    'terminalCovariance',covD,'objective',solD.value(DNLP.objective));
legResults(leg).timing=time;
end

%% 8. Mission totals
utoBuild=arrayfun(@(L)L.timing.UTO.build,legResults);
utoSolve=arrayfun(@(L)L.timing.UTO.solve,legResults);
utoExtract=arrayfun(@(L)L.timing.UTO.interfaceAndExtraction,legResults);
detBuild=arrayfun(@(L)L.timing.standardTO.build,legResults);
detSolve=arrayfun(@(L)L.timing.standardTO.solve,legResults);
detExtract=arrayfun(@(L)L.timing.standardTO.interfaceAndExtraction,legResults);
utoOnline=arrayfun(@(L)L.timing.UTO.onlineTotal,legResults);
detOnline=arrayfun(@(L)L.timing.standardTO.onlineTotal,legResults);
fprintf('\nThree-waypoint timing with one shared parameterized NLP per method\n');
fprintf('                         UTO          Standard TO\n');
fprintf('One-time NLP builds   %10.6f    %10.6f\n',sum(utoBuild),sum(detBuild));
fprintf('IPOPT solve           %10.6f    %10.6f\n',sum(utoSolve),sum(detSolve));
fprintf('Interface/extraction  %10.6f    %10.6f\n',sum(utoExtract),sum(detExtract));
fprintf('PREBUILT ONLINE TOTAL %10.6f    %10.6f\n',sum(utoOnline),sum(detOnline));
fprintf('COLD MISSION TOTAL    %10.6f    %10.6f\n', ...
    sum(utoBuild)+sum(utoOnline),sum(detBuild)+sum(detOnline));

figure('Color','w','Name','Three-waypoint normalized 6-D trajectories');
hold on;hStd=[];hUTO=[];hSigma=[];
for leg=1:3
    for r=1:P.nRegions
        XdLeg=legResults(leg).standardTO.nominalStates(:,:,1,r);
        h=plot3(XdLeg(1,:),XdLeg(2,:),XdLeg(3,:),'r--','LineWidth',1.4);
        if isempty(hStd),hStd=h;end
        Xsigma=legResults(leg).UTO.sigmaStates(:,:,:,r);
        meanBlock=zeros(9,size(Xsigma,2));
        for i=1:nSigma
            meanBlock=meanBlock+w(i)*Xsigma(:,:,i);
            hs=plot3(Xsigma(1,:,i),Xsigma(2,:,i),Xsigma(3,:,i), ...
                'Color',[0.72 0.88 0.72],'LineWidth',0.6);
            if isempty(hSigma),hSigma=hs;end
        end
        hu=plot3(meanBlock(1,:),meanBlock(2,:),meanBlock(3,:), ...
            'Color',[0 0.5 0],'LineWidth',1.8);
        if isempty(hUTO),hUTO=hu;end
    end
end
hBelief=plot3(means(1,:),means(2,:),means(3,:),'ko', ...
    'MarkerFaceColor','k','MarkerSize',5);
hWaypoint=plot3(waypoints(1,:),waypoints(2,:),waypoints(3,:),'kx', ...
    'LineWidth',2,'MarkerSize',9);
grid on;axis equal;xlabel('X (m)');ylabel('Y (m)');zlabel('Z (m)');
title('Three-waypoint TO with parameterized terminal velocity bounds');
legend([hStd,hUTO,hSigma,hBelief,hWaypoint], ...
    {'Standard TO','UTO mean','UTO sigma trajectories', ...
    'Preset belief means','Waypoints'},'Location','best');

%% Three groups: build is charged only on a template's first use
figure('Color','w','Name','Three-waypoint build and solve time');
hold on;
barWidth=0.28;groupOffset=0.18;
standardCenter=(1:3)-groupOffset;
utoCenter=(1:3)+groupOffset;

standardBuildColor=[0.58 0.78 0.93];
standardSolveColor=[0.00 0.4470 0.7410];
utoBuildColor=[1.00 0.72 0.42];
utoSolveColor=[0.8500 0.3250 0.0980];

for k=1:3
    xs=[standardCenter(k)-barWidth/2,standardCenter(k)+barWidth/2];
    xu=[utoCenter(k)-barWidth/2,utoCenter(k)+barWidth/2];

    pStandardBuild=patch([xs(1) xs(2) xs(2) xs(1)], ...
        [0 0 detBuild(k) detBuild(k)],standardBuildColor, ...
        'EdgeColor',[0.2 0.2 0.2]);
    pStandardSolve=patch([xs(1) xs(2) xs(2) xs(1)], ...
        [detBuild(k) detBuild(k) detBuild(k)+detSolve(k) ...
         detBuild(k)+detSolve(k)],standardSolveColor, ...
        'EdgeColor',[0.2 0.2 0.2]);

    pUTOBuild=patch([xu(1) xu(2) xu(2) xu(1)], ...
        [0 0 utoBuild(k) utoBuild(k)],utoBuildColor, ...
        'EdgeColor',[0.2 0.2 0.2]);
    pUTOSolve=patch([xu(1) xu(2) xu(2) xu(1)], ...
        [utoBuild(k) utoBuild(k) utoBuild(k)+utoSolve(k) ...
         utoBuild(k)+utoSolve(k)],utoSolveColor, ...
        'EdgeColor',[0.2 0.2 0.2]);

    if k==1
        hStandardBuild=pStandardBuild;
        hStandardSolve=pStandardSolve;
        hUTOBuild=pUTOBuild;
        hUTOSolve=pUTOSolve;
    else
        set([pStandardBuild,pStandardSolve,pUTOBuild,pUTOSolve], ...
            'HandleVisibility','off');
    end
end

standardDisplayedTotal=detBuild+detSolve;
utoDisplayedTotal=utoBuild+utoSolve;
maximumDisplayed=max([standardDisplayedTotal,utoDisplayedTotal]);
ylim([0,max(1e-6,1.20*maximumDisplayed)]);
xlim([0.5 3.5]);
set(gca,'XTick',1:3,'XTickLabel',{'1','2','3'},'Layer','top');
xlabel('Leg');ylabel('Compute time (s)');
title('Shared-template build charge and IPOPT solve time');grid on;box on;

for k=1:3
    text(standardCenter(k),standardDisplayedTotal(k), ...
        sprintf('Standard\n%.3f s',standardDisplayedTotal(k)), ...
        'HorizontalAlignment','center','VerticalAlignment','bottom', ...
        'FontSize',8);
    text(utoCenter(k),utoDisplayedTotal(k), ...
        sprintf('UTO\n%.3f s',utoDisplayedTotal(k)), ...
        'HorizontalAlignment','center','VerticalAlignment','bottom', ...
        'FontSize',8);
end

legend([hStandardBuild,hStandardSolve,hUTOBuild,hUTOSolve], ...
    {'Standard build','Standard solve','UTO build','UTO solve'}, ...
    'Location','northeast');

results.parameters=struct('settings',P,'means',means, ...
    'initialUncertainCovariances',P0UncertainAll,'waypoints',waypoints, ...
    'velocityReferences',velocityReferences,'weights',w, ...
    'uncertainStateIndices',uncertainStateIndices,'tau',tau,'D',D);
results.legs=legResults;
results.timing=struct('UTO',struct('build',utoBuild,'solve',utoSolve, ...
    'extraction',utoExtract,'online',utoOnline, ...
    'prebuiltOnlineTotal',sum(utoOnline), ...
    'coldMissionTotal',sum(utoBuild)+sum(utoOnline)), ...
    'standardTO',struct('build',detBuild,'solve',detSolve, ...
    'extraction',detExtract,'online',detOnline, ...
    'prebuiltOnlineTotal',sum(detOnline), ...
    'coldMissionTotal',sum(detBuild)+sum(detOnline)), ...
    'templateSetup',setup);
end

function [means,C6,waypoints,vref]=threeWaypointMission(P)
% Parameters copied from UTO_standard_TO_vs_Z2_three_waypoint_validation.
waypoints=[1,1/3,0.20+1/3;2,2/3,0.20+2/3;3,1,1.20].';
vref=zeros(3,3);
vref(:,1)=P.cruiseSpeed*(waypoints(:,2)-waypoints(:,1))/ ...
    norm(waypoints(:,2)-waypoints(:,1));
vref(:,2)=P.cruiseSpeed*(waypoints(:,3)-waypoints(:,2))/ ...
    norm(waypoints(:,3)-waypoints(:,2));

means=zeros(9,3);
means(:,1)=[0;0;0.2;0;0;0;0;0;0];
position2=waypoints(:,1)+[0.012;-0.008;0.006];
velocity2=vref(:,1)+[-0.025;0.012;-0.010];
attitude2=deg2rad([1;-0.5;2]);
means(:,2)=[position2;velocity2;attitude2];
position3=waypoints(:,2)+[-0.015;0.010;-0.008];
velocity3=vref(:,2)+[0.010;-0.020;0.015];
attitude3=deg2rad([-0.5;1;2.5]);
means(:,3)=[position3;velocity3;attitude3];

stdP1=[0.10;0.10;0.14];stdA1=deg2rad([2;2;3]);
stdP2=[0.07;0.08;0.10];stdA2=deg2rad([1.5;1.5;2]);
stdP3=[0.12;0.10;0.16];stdA3=deg2rad([2.5;2;3.5]);
C6(:,:,1)=diag([stdP1;stdA1].^2);
C6(:,:,2)=diag([stdP2;stdA2].^2);
C6(:,:,3)=diag([stdP3;stdA3].^2);
end

function Tf=presetHorizon(mean0,target,isFinal,P)
Tf=norm(target-mean0(1:3))/P.cruiseSpeed+P.horizonBuffer;
if isFinal,Tf=Tf+P.finalStopBuffer;end
Tf=min(max(Tf,P.Tmin),P.Tmax);
end

function O=buildUTOProblem(P,tau,D,w,nSigma)
% One graph for all legs; terminal mode and velocity bounds are parameters.
import casadi.*
opti=Opti();
S0=opti.parameter(9,nSigma);
target=opti.parameter(3,1);
vref=opti.parameter(3,1);
Tf=opti.parameter();
terminalMode=opti.parameter();
terminalVelocityLower=opti.parameter(3,1);
terminalVelocityUpper=opti.parameter(3,1);
duration=Tf/P.nRegions;
[U,X]=transcriptionVariables(opti,nSigma,S0,tau,D,duration,P);
[xBarF,PpositionF,PfullF]=terminalUTStatistics(X,w,P.nRegions,P);
Jcov=trace(P.Qposition*PpositionF);
dvCruise=xBarF(4:6)-vref;
JvelocityCruise=P.terminalVelocityWeight* ...
    (dvCruise.'*P.QterminalVelocity*dvCruise);
JvelocityFinal=P.finalVelocityWeight*(xBarF(4:6).'*xBarF(4:6));
Jvelocity=(1-terminalMode)*JvelocityCruise+terminalMode*JvelocityFinal;
[Jeffort,Jsmooth]=controlCosts(U,Tf,P);
objective=Jcov+Jvelocity+P.controlEffortWeight*Jeffort+ ...
    P.controlSmoothnessWeight*Jsmooth;
opti.subject_to(xBarF(1:3)./P.xScale(1:3)==target./P.xScale(1:3));
opti.subject_to(xBarF(4:6)./P.xScale(4:6)>= ...
    terminalVelocityLower./P.xScale(4:6));
opti.subject_to(xBarF(4:6)./P.xScale(4:6)<= ...
    terminalVelocityUpper./P.xScale(4:6));
opti.minimize(objective);
setIpopt(opti,P);
O=struct('opti',opti,'initial',S0, ...
    'target',target,'velocityReference',vref,'Tf',Tf,'U',{U},'X',{X}, ...
    'terminalMode',terminalMode,'terminalVelocityLower',terminalVelocityLower, ...
    'terminalVelocityUpper',terminalVelocityUpper, ...
    'objective',objective,'Jcov',Jcov,'Jvelocity',Jvelocity, ...
    'JvelocityCruise',JvelocityCruise,'JvelocityFinal',JvelocityFinal, ...
    'terminalMeanVelocity',xBarF(4:6), ...
    'terminalFullCovariance',PfullF,'Jeffort',Jeffort,'Jsmooth',Jsmooth);
end

function O=buildStandardTOProblem(P,tau,D)
% Nominal counterpart with the same shared parameterized graph.
import casadi.*
opti=Opti();
x0=opti.parameter(9,1);
target=opti.parameter(3,1);
vref=opti.parameter(3,1);
Tf=opti.parameter();
terminalMode=opti.parameter();
terminalVelocityLower=opti.parameter(3,1);
terminalVelocityUpper=opti.parameter(3,1);
duration=Tf/P.nRegions;
[U,X]=transcriptionVariables(opti,1,x0,tau,D,duration,P);
terminal=P.Sx*X{1,P.nRegions}(:,end);
dvCruise=terminal(4:6)-vref;
JvelocityCruise=P.terminalVelocityWeight* ...
    (dvCruise.'*P.QterminalVelocity*dvCruise);
JvelocityFinal=P.finalVelocityWeight*(terminal(4:6).'*terminal(4:6));
Jvelocity=(1-terminalMode)*JvelocityCruise+terminalMode*JvelocityFinal;
[Jeffort,Jsmooth]=controlCosts(U,Tf,P);
objective=Jvelocity+P.controlEffortWeight*Jeffort+ ...
    P.controlSmoothnessWeight*Jsmooth;
opti.subject_to(terminal(1:3)./P.xScale(1:3)==target./P.xScale(1:3));
opti.subject_to(terminal(4:6)./P.xScale(4:6)>= ...
    terminalVelocityLower./P.xScale(4:6));
opti.subject_to(terminal(4:6)./P.xScale(4:6)<= ...
    terminalVelocityUpper./P.xScale(4:6));
opti.minimize(objective);
setIpopt(opti,P);
O=struct('opti',opti,'initial',x0, ...
    'target',target,'velocityReference',vref,'Tf',Tf,'U',{U},'X',{X}, ...
    'terminalMode',terminalMode,'terminalVelocityLower',terminalVelocityLower, ...
    'terminalVelocityUpper',terminalVelocityUpper, ...
    'objective',objective,'Jvelocity',Jvelocity, ...
    'JvelocityCruise',JvelocityCruise,'JvelocityFinal',JvelocityFinal, ...
    'Jeffort',Jeffort,'Jsmooth',Jsmooth);
end

function setUTOParameters(O,S0,target,vref,Tf,terminalMode,vLower,vUpper)
O.opti.set_value(O.initial,S0);
O.opti.set_value(O.target,target);
O.opti.set_value(O.velocityReference,vref);
O.opti.set_value(O.Tf,Tf);
O.opti.set_value(O.terminalMode,terminalMode);
O.opti.set_value(O.terminalVelocityLower,vLower);
O.opti.set_value(O.terminalVelocityUpper,vUpper);
end

function setStandardParameters(O,x0,target,vref,Tf,terminalMode,vLower,vUpper)
O.opti.set_value(O.initial,x0);
O.opti.set_value(O.target,target);
O.opti.set_value(O.velocityReference,vref);
O.opti.set_value(O.Tf,Tf);
O.opti.set_value(O.terminalMode,terminalMode);
O.opti.set_value(O.terminalVelocityLower,vLower);
O.opti.set_value(O.terminalVelocityUpper,vUpper);
end

function [U,X]=transcriptionVariables(opti,nSigma,initial,tau,D,duration,P)
% U and X are dimensionless NLP variables: u=Su*U and x=Sx*X.
M=P.nRegions;K=P.nodesPerRegion;U=cell(1,M);X=cell(nSigma,M);
checks=unique([tau,1,linspace(-1,1,P.nControlConstraintPointsPerRegion), ...
    0.5*(tau(1:end-1)+tau(2:end))]);
L=lagrangeValues(tau,checks);
lower=(P.uMin+P.controlInteriorMargin)./P.uScale;
upper=(P.uMax-P.controlInteriorMargin)./P.uScale;
for r=1:M
    U{r}=opti.variable(4,K);
    Ucheck=U{r}*L;nCheck=numel(checks);
    uVector=reshape(Ucheck,4*nCheck,1);
    opti.subject_to(repmat(lower,nCheck,1)<=uVector);
    opti.subject_to(uVector<=repmat(upper,nCheck,1));
    for i=1:nSigma
        X{i,r}=opti.variable(9,K+1);
        addStateBounds(opti,X{i,r},P);
        if r==1
            opti.subject_to(X{i,r}(:,1)==initial(:,i)./P.xScale);
        else
            opti.subject_to(X{i,r}(:,1)==X{i,r-1}(:,end));
        end
        for k=1:K
            xPhysical=P.Sx*X{i,r}(:,k);
            uPhysical=P.Su*U{r}(:,k);
            dynamicsNormalized=P.SxInv*quadDynamics(xPhysical,uPhysical,P);
            opti.subject_to(X{i,r}*D(k,:)'== ...
                (duration/2)*dynamicsNormalized);
        end
    end
end
endpoint=lagrangeValues(tau,1);
for r=1:M-1
    opti.subject_to(U{r}*endpoint==U{r+1}(:,1));
end
end

function addStateBounds(opti,X,P)
n=size(X,2);
velocity=reshape(X(4:6,:),3*n,1);
rollPitch=reshape(X(7:8,:),2*n,1);
velocityLimit=repmat(P.vMax./P.xScale(4:6),n,1);
angleLimit=repmat(P.angleMax./P.xScale(7:8),n,1);
opti.subject_to(-velocityLimit<=velocity);
opti.subject_to(velocity<=velocityLimit);
opti.subject_to(-angleLimit<=rollPitch);
opti.subject_to(rollPitch<=angleLimit);
end

function [meanF,PpositionF,PfullF]=terminalUTStatistics(X,w,M,P)
nSigma=numel(w);meanF=casadi.MX.zeros(9,1);
for i=1:nSigma
    meanF=meanF+w(i)*(P.Sx*X{i,M}(:,end));
end
PfullF=casadi.MX.zeros(9,9);
for i=1:nSigma
    terminalPhysical=P.Sx*X{i,M}(:,end);
    d=terminalPhysical-meanF;
    PfullF=PfullF+w(i)*(d*d.');
end
PpositionF=PfullF(1:3,1:3);
end

function [Jc,Js]=controlCosts(U,Tf,P)
K=size(U{1},2);M=numel(U);[tau,~]=lgrOperators(K);
quadrature=lgrQuadratureWeights(tau);weights=barycentricWeights(tau);
Dc=zeros(K);
for k=1:K,Dc(:,k)=lagrangeDerivativeAtNode(tau,weights,k);end
duration=Tf/M;Jc=0;Js=0;
for r=1:M
    physicalControl=P.Su*U{r};
    rate=(2/duration)*(physicalControl*Dc);
    for k=1:K
        du=physicalControl(:,k)-P.uHover;
        Jc=Jc+(duration/2)*quadrature(k)*(du.'*P.Ru*du);
        Js=Js+(duration/2)*quadrature(k)*(rate(:,k).'*P.Rdu*rate(:,k));
    end
end
end

function setInitialGuess(opti,U,X,X0,target,vref,Tf,tau,P)
opti.set_initial(opti.lam_g,zeros(opti.ng,1));
for r=1:P.nRegions
    opti.set_initial(U{r}, ...
        repmat(P.uHover./P.uScale,1,P.nodesPerRegion));
    for i=1:size(X0,2)
        physicalGuess=guessBlock(X0(:,i),target,vref,Tf,r,tau,P);
        opti.set_initial(X{i,r},physicalGuess./P.xScale);
    end
end
end

function G=guessBlock(x0,target,vref,Tf,r,tau,P)
K=P.nodesPerRegion;G=zeros(9,K+1);
for k=1:K
    q=((r-1)+(tau(k)+1)/2)/P.nRegions;
    G(:,k)=trajectoryGuess(x0,target,vref,q,Tf);
end
G(:,end)=trajectoryGuess(x0,target,vref,r/P.nRegions,Tf);
end

function x=trajectoryGuess(x0,target,vTarget,q,T)
h00=2*q^3-3*q^2+1;h10=q^3-2*q^2+q;
h01=-2*q^3+3*q^2;h11=q^3-q^2;
dh00=6*q^2-6*q;dh10=3*q^2-4*q+1;
dh01=-6*q^2+6*q;dh11=3*q^2-2*q;
x=x0;
x(1:3)=h00*x0(1:3)+h10*T*x0(4:6)+h01*target+h11*T*vTarget;
x(4:6)=(dh00*x0(1:3)+dh10*T*x0(4:6)+dh01*target+dh11*T*vTarget)/T;
x(7:8)=(1-q)*x0(7:8);
end

function [Uvalue,Xvalue]=extractLGRSolution(sol,U,X,P)
M=P.nRegions;K=P.nodesPerRegion;nSigma=size(X,1);
Uvalue=zeros(4,K,M);Xvalue=zeros(9,K+1,nSigma,M);
for r=1:M
    Uvalue(:,:,r)=P.Su*sol.value(U{r});
    for i=1:nSigma
        Xvalue(:,:,i,r)=P.Sx*sol.value(X{i,r});
    end
end
end

function terminal=propagateLGRControlTerminal(x0,U,tau,Tf,P)
duration=Tf/P.nRegions;x=x0;
options=odeset('RelTol',1e-8,'AbsTol',1e-10,'MaxStep',duration/50);
for r=1:P.nRegions
    a=(r-1)*duration;b=r*duration;
    rhs=@(t,z)quadDynamics(z,localLGRControl(t,U,tau,r,duration),P);
    [~,trajectory]=ode45(rhs,[a b],x,options);
    x=trajectory(end,:).';
end
x(7:9)=wrapAngles(x(7:9));terminal=x;
end

function u=localLGRControl(t,U,tau,r,duration)
a=(r-1)*duration;q=2*(t-a)/duration-1;
u=barycentricValueDerivative(tau,U(:,:,r),min(max(q,-1),1));
end

function [meanState,C]=weightedTerminalStatistics(X,w)
meanState=X*w(:);
for a=7:9
    meanState(a)=atan2(sum(w.*sin(X(a,:))),sum(w.*cos(X(a,:))));
end
C=zeros(9);
for i=1:numel(w)
    d=X(:,i)-meanState;d(7:9)=wrapAngles(d(7:9));
    C=C+w(i)*(d*d.');
end
C=0.5*(C+C.');
end

function dx=quadDynamics(x,u,P)
phi=x(7);theta=x(8);psi=x(9);
R=[cos(psi)*cos(theta), ...
   cos(psi)*sin(theta)*sin(phi)-sin(psi)*cos(phi), ...
   cos(psi)*sin(theta)*cos(phi)+sin(psi)*sin(phi); ...
   sin(psi)*cos(theta), ...
   sin(psi)*sin(theta)*sin(phi)+cos(psi)*cos(phi), ...
   sin(psi)*sin(theta)*cos(phi)-cos(psi)*sin(phi); ...
   -sin(theta),cos(theta)*sin(phi),cos(theta)*cos(phi)];
acceleration=R*[0;0;u(1)]-[0;0;P.g]-P.drag*x(4:6);
dx=[x(4:6);acceleration;(u(2)-phi)/P.tauAtt; ...
    (u(3)-theta)/P.tauAtt;u(4)];
end

function setIpopt(opti,P)
opts=struct('expand',true,'detect_simple_bounds',true,'print_time',false);
opts.ipopt.print_level=0;opts.ipopt.max_iter=P.ipoptMaxIter;
opts.ipopt.tol=P.ipoptTol;opts.ipopt.acceptable_tol=P.ipoptAcceptableTol;
opts.ipopt.hessian_approximation='exact';
opts.ipopt.nlp_scaling_method='none';
opti.solver('ipopt',opts);
end

function [V,w]=simplexSigmaPoints(d)
n=d+1;Q=null(ones(1,n));V=sqrt(n)*Q.';w=ones(1,n)/n;
end

function [value,derivative]=barycentricValueDerivative(nodes,data,query)
nodes=nodes(:).';query=query(:).';weights=barycentricWeights(nodes);
value=zeros(size(data,1),numel(query));derivative=zeros(size(value));
for k=1:numel(query)
    difference=query(k)-nodes;[nearest,index]=min(abs(difference));
    if nearest<=32*eps(max(1,abs(query(k))))
        value(:,k)=data(:,index);
        if nargout>1,derivative(:,k)=data*lagrangeDerivativeAtNode(nodes,weights,index);end
    else
        inverse=weights./difference;denominator=sum(inverse);
        value(:,k)=(data*inverse.')/denominator;
        if nargout>1
            inverseSquared=weights./difference.^2;
            derivative(:,k)=(value(:,k)*sum(inverseSquared)- ...
                data*inverseSquared.')/denominator;
        end
    end
end
end

function weights=barycentricWeights(nodes)
n=numel(nodes);weights=ones(1,n);
for j=1:n,weights(j)=1/prod(nodes(j)-nodes([1:j-1,j+1:n]));end
weights=weights/max(abs(weights));
end

function column=lagrangeDerivativeAtNode(nodes,weights,index)
n=numel(nodes);column=zeros(n,1);
for j=1:n
    if j~=index,column(j)=weights(j)/(weights(index)*(nodes(index)-nodes(j)));end
end
column(index)=-sum(column);
end

function values=lagrangeValues(nodes,query)
values=barycentricValueDerivative(nodes,eye(numel(nodes)),query);
end

function weights=lgrQuadratureWeights(tau)
K=numel(tau);V=zeros(K);moments=zeros(K,1);
for degree=0:K-1
    V(degree+1,:)=tau.^degree;
    if mod(degree,2)==0,moments(degree+1)=2/(degree+1);end
end
weights=V\moments;
end

function [tau,D]=lgrOperators(K)
if K==1
    tau=-1;
else
    m=K-1;a=0;b=1;diagonal=zeros(m,1);offDiagonal=zeros(m-1,1);
    for n=0:m-1
        diagonal(n+1)=(b^2-a^2)/((2*n+a+b)*(2*n+a+b+2));
    end
    for n=1:m-1
        offDiagonal(n)=2/(2*n+a+b)*sqrt(n*(n+a)*(n+b)*(n+a+b)/ ...
            ((2*n+a+b-1)*(2*n+a+b+1)));
    end
    tau=[-1;sort(eig(diag(diagonal)+diag(offDiagonal,1)+ ...
        diag(offDiagonal,-1)))]';
end
s=[tau,1];weights=barycentricWeights(s);F=zeros(K+1);
for k=1:K+1
    for j=1:K+1
        if j~=k,F(k,j)=weights(j)/(weights(k)*(s(k)-s(j)));end
    end
    F(k,k)=-sum(F(k,:));
end
D=F(1:K,:);
end

function value=getSolverWallTime(stats)
value=NaN;
if isfield(stats,'t_wall_total'),value=stats.t_wall_total;
elseif isfield(stats,'t_proc_total'),value=stats.t_proc_total;end
end

function value=getIterationCount(stats)
value=-1;
if isfield(stats,'iter_count'),value=stats.iter_count;end
end

function a=wrapAngles(a)
a=atan2(sin(a),cos(a));
end

function hPlot=plotCovarianceEllipse(mu,P,dims,colorValue,lineStyle,markerStyle)
P2=0.5*(P(dims,dims)+P(dims,dims).');[V,D]=eig(P2);
d=max(diag(D),1e-14);theta=linspace(0,2*pi,121);
ellipse=V*diag(sqrt(5.991*d))*[cos(theta);sin(theta)]+mu(dims);
hPlot=plot(ellipse(1,:),ellipse(2,:),'Color',colorValue, ...
    'LineStyle',lineStyle,'LineWidth',1.6,'Marker',markerStyle, ...
    'MarkerIndices',1:6:numel(theta),'MarkerSize',3);
end
